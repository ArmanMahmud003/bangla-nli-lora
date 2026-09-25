"""Train LoRA and full fine-tuning of BanglaBERT on XNLI-bn under a matched budget.

    python src/train.py --out runs_out --fractions 0.01 --seeds 42

Each (method, fraction, seed) run trains for a fixed number of epochs with no early
stopping, keeps the epoch with the best validation macro-F1, scores the test split
once with that model, and writes:

    <out>/runs/<run_id>.json        manifest: settings, metrics, efficiency, versions
    <out>/curves/<run_id>.json      per-epoch validation metrics
    <out>/preds/<run_id>_test.npz   test logits and labels
    <out>/results.csv               one row per manifest

Runs that already have a manifest are skipped unless --force is given.
"""
import argparse
import gc
import glob
import json
import math
import os
import shutil
import time

import numpy as np
import pandas as pd
import peft
import torch
import transformers
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (AutoModelForSequenceClassification, Trainer,
                          TrainingArguments, default_data_collator, set_seed)

from data import compute_metrics, load_tokenized

BASE_MODEL = "csebuetnlp/banglabert"
NUM_LABELS = 3
BATCH_SIZE = 16
EVAL_BATCH = 64
EPOCHS = 8
LR = {"lora": 2e-4, "full_ft": 2e-5}
LORA_CFG = dict(r=8, lora_alpha=16, lora_dropout=0.1, target_modules=["query", "value"])
# Full fine-tuned weights (~443 MB) are measured for the efficiency table, then deleted.
KEEP_MODEL = {"lora": True, "full_ft": False}
# A best validation macro-F1 below this means the run diverged; no manifest is written.
CHANCE_F1 = 0.40
KEEP_COLUMNS = ["input_ids", "attention_mask", "token_type_ids", "label"]


def gpu_name():
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


def strip(ds):
    return ds.remove_columns([c for c in ds.column_names if c not in KEEP_COLUMNS])


def dir_mb(path):
    return round(sum(os.path.getsize(os.path.join(r, f))
                     for r, _, fs in os.walk(path) for f in fs) / 1e6, 2)


def build_model(method, seed):
    set_seed(seed)  # before construction: classifier head and LoRA A/B init
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=NUM_LABELS)
    if method == "lora":
        model = get_peft_model(model, LoraConfig(task_type=TaskType.SEQ_CLS, **LORA_CFG))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return model, trainable, total


def reload_drift(save_dir, method, test_ds, expected, precision):
    """Max absolute difference between test logits of the saved-and-reloaded
    model and the in-memory model."""
    if method == "lora":
        base = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL, num_labels=NUM_LABELS)
        model = PeftModel.from_pretrained(base, save_dir)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(save_dir)
    args = TrainingArguments(output_dir=os.path.join(save_dir, "_verify"),
                             per_device_eval_batch_size=EVAL_BATCH, report_to="none",
                             fp16=precision == "fp16", bf16=precision == "bf16")
    logits = Trainer(model=model.eval(), args=args,
                     data_collator=default_data_collator).predict(test_ds).predictions
    del model
    gc.collect()
    return float(np.max(np.abs(logits.astype(np.float32) - expected.astype(np.float32))))


def run_one(data, out, scratch, method, fraction, seed, precision="fp16", force=False):
    run_id = f"{method}_frac{fraction:g}_seed{seed}"
    manifest_path = f"{out}/runs/{run_id}.json"
    if os.path.exists(manifest_path) and not force:
        print(f"[skip] {run_id}")
        return json.load(open(manifest_path))

    n_train = int(len(data["train"]) * fraction)
    # Depends only on the seed: both methods see the same subset, and subsets nest
    # across fractions.
    train_ds = strip(data["train"].shuffle(seed=seed).select(range(n_train)))
    dev_ds, test_ds = strip(data["validation"]), strip(data["test"])

    model, trainable, total = build_model(method, seed)
    print(f"[run] {run_id}: n_train={n_train}, trainable {trainable:,}/{total:,}")

    ckpt_dir = f"{scratch}/{run_id}"
    shutil.rmtree(ckpt_dir, ignore_errors=True)
    args = TrainingArguments(
        output_dir=ckpt_dir,
        num_train_epochs=EPOCHS,
        learning_rate=LR[method],
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=1,
        seed=seed,
        data_seed=seed,
        fp16=precision == "fp16",
        bf16=precision == "bf16",
        logging_steps=max(10, math.ceil(n_train / BATCH_SIZE) // 2),
        report_to="none",
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=dev_ds,
                      compute_metrics=compute_metrics, data_collator=default_data_collator)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    train_out = trainer.train()
    wall = time.time() - t0
    peak_mb = (round(torch.cuda.max_memory_allocated() / 1e6, 1)
               if torch.cuda.is_available() else None)

    curve = [e for e in trainer.state.log_history if "eval_f1_macro" in e]
    best = max(curve, key=lambda e: e["eval_f1_macro"])
    json.dump(curve, open(f"{out}/curves/{run_id}.json", "w"), indent=2)
    if best["eval_f1_macro"] < CHANCE_F1:
        raise RuntimeError(f"{run_id}: best dev macro-F1 {best['eval_f1_macro']:.4f} "
                           f"is at chance level; run diverged")

    # The restored model must reproduce the best curve point on dev.
    dev = trainer.evaluate(dev_ds, metric_key_prefix="dev")
    test = trainer.predict(test_ds, metric_key_prefix="test")
    np.savez_compressed(f"{out}/preds/{run_id}_test.npz",
                        logits=test.predictions.astype(np.float16),
                        labels=np.asarray(test.label_ids))

    save_dir = (f"{out}/models/{run_id}" if KEEP_MODEL[method]
                else f"{scratch}/_size/{run_id}")
    shutil.rmtree(save_dir, ignore_errors=True)
    trainer.save_model(save_dir)
    saved_mb = dir_mb(save_dir)
    drift = reload_drift(save_dir, method, test_ds, test.predictions, precision)
    if not KEEP_MODEL[method]:
        shutil.rmtree(save_dir, ignore_errors=True)

    manifest = {
        "run_id": run_id, "method": method, "fraction": fraction, "seed": seed,
        "n_train": n_train, "epoch_ceiling": EPOCHS,
        "best_epoch": float(best["epoch"]), "best_step": int(best.get("step", 0)),
        "evals_ran": len(curve), "evals_possible": EPOCHS,
        "early_stopped": len(curve) < EPOCHS,
        "ceiling_hit": abs(float(best["epoch"]) - EPOCHS) < 1e-6,
        "patience_epochs": None, "evals_per_epoch": 1,
        "learning_rate": LR[method], "batch_size": BATCH_SIZE, "precision": precision,
        "dev_accuracy": float(dev["dev_accuracy"]),
        "dev_f1_macro": float(dev["dev_f1_macro"]),
        "dev_f1_from_curve": float(best["eval_f1_macro"]),
        "dev_reload_drift": float(abs(dev["dev_f1_macro"] - best["eval_f1_macro"])),
        "test_accuracy": float(test.metrics["test_accuracy"]),
        "test_f1_macro": float(test.metrics["test_f1_macro"]),
        "trainable_params": int(trainable), "total_params": int(total),
        "pct_trainable": round(100 * trainable / total, 4),
        "train_runtime_s": round(wall, 1),
        "train_samples_per_s": round(train_out.metrics.get("train_samples_per_second", 0), 2),
        "peak_gpu_mem_mb": peak_mb, "saved_model_mb": saved_mb,
        "best_model_kept": KEEP_MODEL[method],
        "reload_max_logit_drift": round(drift, 8),
        "gpu": gpu_name(), "transformers": transformers.__version__,
        "peft": peft.__version__, "torch": torch.__version__,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    json.dump(manifest, open(manifest_path, "w"), indent=2)
    print(f"  best epoch {manifest['best_epoch']:g}, dev F1 {manifest['dev_f1_macro']:.4f}, "
          f"test F1 {manifest['test_f1_macro']:.4f}, {wall / 60:.1f} min")

    del model, trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    shutil.rmtree(ckpt_dir, ignore_errors=True)
    return manifest


def write_results_csv(out):
    rows = [json.load(open(p)) for p in sorted(glob.glob(f"{out}/runs/*.json"))]
    if rows:
        (pd.DataFrame(rows).sort_values(["fraction", "seed", "method"])
         .to_csv(f"{out}/results.csv", index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="runs_out")
    ap.add_argument("--scratch", default="/tmp/ckpt",
                    help="checkpoint directory, wiped after each run")
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.01, 0.05, 0.10])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 13, 21, 33])
    ap.add_argument("--methods", nargs="+", default=["full_ft", "lora"],
                    choices=["full_ft", "lora"])
    ap.add_argument("--precision", default="fp16", choices=["fp16", "bf16", "fp32"],
                    help="all published runs used fp16")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    for d in ("runs", "curves", "preds", "models"):
        os.makedirs(f"{a.out}/{d}", exist_ok=True)
    data = load_tokenized()
    for f in sorted(a.fractions):
        for s in a.seeds:
            for m in a.methods:
                run_one(data, a.out, a.scratch, m, f, s, a.precision, a.force)
                write_results_csv(a.out)


if __name__ == "__main__":
    main()
