"""Convergence, efficiency, prediction agreement and figures for the four-method runs.

    python src/analyze_v2_extra.py      # writes results/v2/{convergence,efficiency,agreement}_v2.csv
                                        # and figures/v2_*.png

Uses the same run selection as src/analyze_v2.py (all XNLI-bn runs on an A100).
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import analyze_v2 as v2

ROOT, OUT = v2.ROOT, v2.OUT
FIG = os.path.join(ROOT, "..", "figures")
NAME = {"full_ft": "Full FT", "lora": "LoRA", "dora": "DoRA", "ia3": "IA3"}
COLOR = {"full_ft": "#2a78d6", "lora": "#eb6834", "dora": "#1baf7a", "ia3": "#eda100"}   # validated, fixed order
MARKER = {"full_ft": "o", "lora": "s", "dora": "^", "ia3": "D"}


def manifest(method, frac, seed, dataset):
    rid = f"{method}_frac{frac}_seed{seed}"
    if dataset == "bnli":
        path = f"{ROOT}/bnli/runs/{rid}.json"
    elif method in ("dora", "ia3") or seed in v2.T4_SEEDS.get(frac, []):
        path = f"{ROOT}/peft/runs/{rid}.json"
    else:
        path = f"{ROOT}/runs/{rid}.json"
    m = json.load(open(path))
    assert "A100" in m["gpu"], f"{path} is not an A100 run"
    return m


def all_manifests():
    rows = []
    for ds, fracs in (("xnli_bn", v2.FRACTIONS), ("bnli", [1.0])):
        for f in fracs:
            for s in v2.SEEDS:
                for meth in v2.METHODS:
                    m = manifest(meth, f, s, ds)
                    rows.append(dict(dataset=ds, method=meth, fraction=f, seed=s,
                                     best_epoch=m["best_epoch"], ceiling_hit=m["best_epoch"] >= m["epoch_ceiling"],
                                     trainable=m["trainable_params"], total=m["total_params"],
                                     saved_mb=m["saved_model_mb"], peak_mb=m["peak_gpu_mem_mb"],
                                     train_s=m["train_runtime_s"], samples_s=m["train_samples_per_s"]))
    return pd.DataFrame(rows)


def convergence(df):
    rows = []
    for (ds, f), g in df.groupby(["dataset", "fraction"]):
        ref = g[g.method == "full_ft"].set_index("seed").best_epoch
        for meth, gm in g.groupby("method"):
            be = gm.set_index("seed").best_epoch.loc[v2.SEEDS]
            r = dict(dataset=ds, fraction=f, method=meth,
                     best_epochs=" ".join(str(int(x)) for x in sorted(be)),
                     mean_best_epoch=be.mean(), ceiling_hits=int(gm.ceiling_hit.sum()))
            if meth != "full_ft":
                d = (be - ref.loc[v2.SEEDS]).values
                r["minus_full_ft"] = d.mean()
                r["p_paired_t_descriptive"] = float(stats.ttest_1samp(d, 0).pvalue) if d.std() > 0 else np.nan
            rows.append(r)
    return pd.DataFrame(rows)


def efficiency(df):
    g = df.groupby(["dataset", "fraction", "method"])
    return pd.DataFrame(dict(trainable=g.trainable.first(), pct_trainable=(g.trainable.first() / g.total.first() * 100).round(3),
                             saved_mb=g.saved_mb.first(), peak_mb=g.peak_mb.mean().round(1),
                             train_min=(g.train_s.mean() / 60).round(2),
                             samples_per_s=g.samples_s.mean().round(1))).reset_index()


def agreement():
    rows = []
    pairs = [(a, b) for i, a in enumerate(v2.METHODS) for b in v2.METHODS[i + 1:]]
    for ev, (fracs, path, _) in v2.EVALS.items():
        for f in fracs:
            preds = {(m, s): np.load(path(m, f, s))["logits"].argmax(1) for m in v2.METHODS for s in v2.SEEDS}
            for a, b in pairs:
                agr = [np.mean(preds[(a, s)] == preds[(b, s)]) for s in v2.SEEDS]
                rows.append(dict(eval=ev, fraction=f, a=a, b=b, agreement_mean=np.mean(agr), agreement_sd=np.std(agr, ddof=1)))
            for m in v2.METHODS:   # same method, different seeds: the baseline for "how much do predictions move"
                agr = [np.mean(preds[(m, s)] == preds[(m, t)]) for i, s in enumerate(v2.SEEDS) for t in v2.SEEDS[i + 1:]]
                rows.append(dict(eval=ev, fraction=f, a=m, b=m + " (other seed)", agreement_mean=np.mean(agr), agreement_sd=np.std(agr, ddof=1)))
    return pd.DataFrame(rows)


def figures():
    per_run = pd.read_csv(f"{OUT}/per_run_v2.csv")
    os.makedirs(FIG, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#888888", "axes.labelcolor": "#333333",
                         "xtick.color": "#555555", "ytick.color": "#555555"})
    panels = [("xnli_to_xnli", "macro_f1", "Test macro-F1 (higher is better)"),
              ("xnli_to_xnli", "ECE", "Test ECE, 15 bins (lower is better)"),
              ("xnli_to_bnli", "macro_f1", "BNLI test macro-F1 (cross-dataset)"),
              ("xnli_to_bnli", "ECE", "BNLI test ECE (cross-dataset)")]
    for fname, chosen in (("v2_xnli_f1_ece.png", panels[:2]), ("v2_cross_dataset_f1_ece.png", panels[2:])):
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
        for ax, (ev, met, title) in zip(axes, chosen):
            sub = per_run[per_run["eval"] == ev]
            for i, m in enumerate(v2.METHODS):
                g = sub[sub.method == m].groupby("fraction")[met].agg(["mean", "std"])
                x = np.arange(len(g)) + (i - 1.5) * 0.06            # small dodge so error bars don't overlap
                ax.errorbar(x, g["mean"], yerr=g["std"], color=COLOR[m], marker=MARKER[m], markersize=5,
                            linewidth=2, capsize=2, label=NAME[m])
            ax.set_xticks(range(3), ["1%", "5%", "10%"])
            ax.set_xlim(-0.3, 2.3)
            ax.set_xlabel("XNLI-bn training data")
            ax.set_title(title, fontsize=9, loc="left")
            ax.grid(axis="y", color="#e5e5e5", linewidth=0.6)
        axes[0].legend(frameon=False, fontsize=7, loc="lower right" if chosen[0][1] == "macro_f1" else "upper right")
        fig.text(0.01, 0.01, "Mean ± SD over 5 seeds; all runs on an A100.", fontsize=7, color="#555555")
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        fig.savefig(os.path.join(FIG, fname), dpi=200)
        plt.close(fig)


def main():
    df = all_manifests()
    convergence(df).round(4).to_csv(f"{OUT}/convergence_v2.csv", index=False)
    efficiency(df).to_csv(f"{OUT}/efficiency_v2.csv", index=False)
    agreement().round(4).to_csv(f"{OUT}/agreement_v2.csv", index=False)
    figures()
    print("written: convergence_v2.csv, efficiency_v2.csv, agreement_v2.csv, figures/v2_*.png")


if __name__ == "__main__":
    main()
