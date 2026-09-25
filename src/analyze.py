"""Every number in the README, recomputed from results/. No GPU needed.

    python src/analyze.py                  # tables to stdout, figures to figures/
    python src/analyze.py --margin 0.02    # TOST at another margin (default 0.01)

Reads runs/*.json (manifests), preds/*_test.npz (test logits and labels),
curves/*.json (per-epoch dev curves) and superseded_runs.csv (the six earlier runs
that A100 re-runs replaced).

Calibration differences get two intervals, because they answer different questions:
  seed-level  paired t over the five seeds. Includes training variability, so it is
              the one that says whether retraining would give the same answer.
  item-level  paired bootstrap over test items with the five trained models held
              fixed. It only covers test-set sampling noise for these models.

Only test logits are saved, so temperature scaling is 5-fold cross-validated on the
test set: T is fitted on four folds and applied to the fifth.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar
from sklearn.metrics import f1_score

METHODS = ("full_ft", "lora")
LABEL = {"full_ft": "Full fine-tuning", "lora": "LoRA"}
N_BINS = 15
BOOT_N = 10000
ITEM_BOOT_N = 2000
TOST_MARGIN = 0.01   # one macro-F1 point; set after data collection, not pre-registered


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def ece(p, y, n_bins=N_BINS):
    """Top-label expected calibration error. Equal-width bins span the attainable
    confidence range [1/C, 1]; with 3 classes the top probability is never below 1/3."""
    conf, correct = p.max(1), p.argmax(1) == y
    edges = np.linspace(1.0 / p.shape[1], 1.0, n_bins + 1)
    edges[0], edges[-1] = 0.0, 1.0 + 1e-9
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return total


def nll(p, y):
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1.0)).mean())


def brier(p, y):
    return float(((p - np.eye(p.shape[1])[y]) ** 2).sum(1).mean())


CALIBRATION = {"ECE": ece, "Brier": brier, "NLL": nll}


def fit_temperature(logits, y):
    return minimize_scalar(lambda t: nll(softmax(logits / t), y),
                           bounds=(0.05, 10.0), method="bounded").x


def cv_temperature(logits, y, seed, k=5):
    """Temperature-scaled probabilities, each fold scaled by a T fitted on the
    other k-1 folds. Returns (probs, mean T)."""
    folds = np.array_split(np.random.default_rng(seed).permutation(len(y)), k)
    out, temps = np.zeros_like(logits), []
    for i, test in enumerate(folds):
        train = np.concatenate([f for j, f in enumerate(folds) if j != i])
        t = fit_temperature(logits[train], y[train])
        temps.append(t)
        out[test] = softmax(logits[test] / t)
    return out, float(np.mean(temps))


def item_bootstrap_gap(pairs, metric, rng, n_boot=ITEM_BOOT_N):
    """95% CI on the seed-averaged LoRA - full difference in `metric`, resampling
    test items (paired across methods and seeds) with the models held fixed."""
    y = pairs[0][2]
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        boot.append(np.mean([metric(pl[idx], yy[idx]) - metric(pf[idx], yy[idx])
                             for pf, pl, yy in pairs]))
    return np.percentile(boot, [2.5, 97.5])


def paired_t_ci(d, level=0.95):
    d = np.asarray(d)
    half = stats.t.ppf(0.5 + level / 2, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return d.mean() - half, d.mean() + half


def holm(pvals):
    order = np.argsort(pvals)
    adj = np.empty(len(pvals))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def mcnemar_exact(right_a, right_b):
    b = int((right_a & ~right_b).sum())
    c = int((~right_a & right_b).sum())
    return b, c, stats.binomtest(b, b + c, 0.5).pvalue if b + c else 1.0


def gap_line(d, margin):
    lo, hi = paired_t_ci(d)
    lo90, hi90 = paired_t_ci(d, 0.90)
    se = d.std(ddof=1) / np.sqrt(len(d))
    p_tost = max(1 - stats.t.cdf((d.mean() + margin) / se, len(d) - 1),
                 stats.t.cdf((d.mean() - margin) / se, len(d) - 1))
    return (f"gap {d.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
            f"p={stats.ttest_rel(d, np.zeros(len(d))).pvalue:.3f}  "
            f"90% CI [{lo90:+.4f}, {hi90:+.4f}]  "
            f"TOST +/-{margin:g} p={p_tost:.6f}")


def sensitivity(art, wide, margin):
    """Re-run the macro-F1 comparison with the six superseded second-pass runs in
    place of their A100 replacements. Only test macro-F1 survives for those runs (no
    logits), so this covers the accuracy table only."""
    path = f"{art}/superseded_runs.csv"
    if not os.path.exists(path):
        return
    old = pd.read_csv(path)
    print("\n=== replacement check: superseded run vs its A100 re-run (test macro-F1) ===")
    alt = wide["f1"].copy()
    for _, r in old.iterrows():
        new = alt.loc[(r.fraction, r.seed), r.method]
        print(f"{r.run_id:26s} {r.gpu:9s} (transformers {r.transformers}) {r.test_f1_macro:.4f}"
              f" -> A100 {new:.4f}  diff {new - r.test_f1_macro:+.4f}")
        alt.loc[(r.fraction, r.seed), r.method] = r.test_f1_macro
    print("pair gaps (LoRA - full), superseded vs delivered:")
    for (f, s), g in old.groupby(["fraction", "seed"]):
        before = alt.loc[(f, s), "lora"] - alt.loc[(f, s), "full_ft"]
        after = wide.loc[(f, s), ("f1", "lora")] - wide.loc[(f, s), ("f1", "full_ft")]
        print(f"  {f:<5g} seed {s:<3} {before:+.4f} -> {after:+.4f}")
    print("macro-F1 comparison with the superseded runs substituted back:")
    for f in sorted(old.fraction.unique()):
        d = alt.loc[f, "lora"] - alt.loc[f, "full_ft"]
        print(f"  {f:<5g} full {alt.loc[f, 'full_ft'].mean():.4f}  lora "
              f"{alt.loc[f, 'lora'].mean():.4f}  " + gap_line(d, margin))


def load(art, run_id):
    f = np.load(f"{art}/preds/{run_id}_test.npz")
    return f["logits"].astype(np.float64), f["labels"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--artifacts", default="results")
    ap.add_argument("--figures", default="figures")
    ap.add_argument("--margin", type=float, default=TOST_MARGIN,
                    help="TOST equivalence margin on macro-F1 (default 0.01)")
    args = ap.parse_args()
    art = args.artifacts

    runs = pd.DataFrame([json.load(open(p)) for p in sorted(glob.glob(f"{art}/runs/*.json"))])
    if runs.empty:
        raise SystemExit(f"no manifests under {art}/runs")
    fractions = sorted(runs.fraction.unique())
    seeds = sorted(runs.seed.unique())
    print(f"{len(runs)} runs, fractions {fractions}, seeds {seeds}")
    print("\nruns per GPU and library version:")
    print(runs.groupby(["gpu", "transformers", "peft", "precision"]).size().to_string())

    # ---- per-run test metrics from the logits --------------------------------
    rows, preds = [], {}
    for _, r in runs.iterrows():
        logits, y = load(art, r.run_id)
        p = softmax(logits)
        p_ts, temp = cv_temperature(logits, y, seed=int(r.seed))
        preds[r.run_id] = (logits, p, y, p_ts)
        rows.append(dict(run_id=r.run_id, method=r.method, fraction=r.fraction, seed=r.seed,
                         gpu=r.gpu, best_epoch=r.best_epoch, ceiling_hit=r.ceiling_hit,
                         f1=f1_score(y, p.argmax(1), average="macro"),
                         acc=(p.argmax(1) == y).mean(), conf=p.max(1).mean(),
                         ECE=ece(p, y), Brier=brier(p, y), NLL=nll(p, y),
                         ECE_ts=ece(p_ts, y), T=temp))
    m = pd.DataFrame(rows)
    wide = m.pivot_table(index=["fraction", "seed"], columns="method",
                         values=["f1", "ECE", "Brier", "NLL", "ECE_ts", "best_epoch"])

    split = runs.groupby(["fraction", "seed"]).gpu.nunique()
    print(f"(fraction, seed) pairs split across GPUs: {int((split > 1).sum())}")

    # ---- accuracy + TOST -----------------------------------------------------
    print(f"\n=== test macro-F1 (TOST margin +/-{args.margin:g}, p < 0.05 = equivalent) ===")
    for f in fractions:
        a, b = wide.loc[f, ("f1", "full_ft")], wide.loc[f, ("f1", "lora")]
        print(f"{f:<5g} full {a.mean():.4f} (sd {a.std():.4f})  lora {b.mean():.4f} "
              f"(sd {b.std():.4f})  " + gap_line(b - a, args.margin))

    sensitivity(art, wide, args.margin)

    # ---- per-seed McNemar with Holm -----------------------------------------
    print("\n=== per-seed McNemar on accuracy, Holm-corrected over all pairs ===")
    tests = []
    for f in fractions:
        for s in seeds:
            _, pf, y, _ = preds[f"full_ft_frac{f:g}_seed{s}"]
            _, pl, _, _ = preds[f"lora_frac{f:g}_seed{s}"]
            b, c, p = mcnemar_exact(pf.argmax(1) == y, pl.argmax(1) == y)
            tests.append((f, s, b, c, p, (pf.argmax(1) == pl.argmax(1)).mean()))
    adj = holm(np.array([t[4] for t in tests]))
    for (f, s, b, c, p, _), pa in zip(tests, adj):
        flag = "  <- p<0.05" if p < 0.05 else ""
        print(f"{f:<5g} seed {s:<3} full-only right {b:4d}  lora-only right {c:4d}  "
              f"p={p:.4f}  holm={pa:.4f}{flag}")
    print(f"{sum(t[4] < 0.05 for t in tests)} of {len(tests)} raw p<0.05; "
          f"{int((adj < 0.05).sum())} survive Holm")

    print("\n=== prediction agreement (same seed) ===")
    agree = pd.DataFrame(tests, columns=["fraction", "seed", "b", "c", "p", "agree"])
    print(agree.groupby("fraction")[["agree", "b", "c"]]
          .agg({"agree": "mean", "b": "sum", "c": "sum"}).round(4).to_string())

    # ---- calibration ---------------------------------------------------------
    rng = np.random.default_rng(20260910)
    print(f"\n=== calibration, LoRA - full (negative = LoRA better; ECE {N_BINS} bins) ===")
    print("seed-level: paired t over seeds (includes training variability)")
    print("item-level: paired bootstrap over test items, models held fixed")
    for f in fractions:
        pairs_raw = [(preds[f"full_ft_frac{f:g}_seed{s}"][1], preds[f"lora_frac{f:g}_seed{s}"][1],
                      preds[f"full_ft_frac{f:g}_seed{s}"][2]) for s in seeds]
        pairs_ts = [(preds[f"full_ft_frac{f:g}_seed{s}"][3], preds[f"lora_frac{f:g}_seed{s}"][3],
                     preds[f"full_ft_frac{f:g}_seed{s}"][2]) for s in seeds]
        sub = m[m.fraction == f].groupby("method")[["conf", "acc", "T"]].mean()
        print(f"{f:g}: mean confidence full {sub.loc['full_ft', 'conf']:.3f} / lora "
              f"{sub.loc['lora', 'conf']:.3f} at accuracy {sub.loc['full_ft', 'acc']:.3f} / "
              f"{sub.loc['lora', 'acc']:.3f}; fitted T full {sub.loc['full_ft', 'T']:.2f}, "
              f"lora {sub.loc['lora', 'T']:.2f}")
        for name, col, pairs, fn in (("ECE", "ECE", pairs_raw, ece),
                                     ("Brier", "Brier", pairs_raw, brier),
                                     ("NLL", "NLL", pairs_raw, nll),
                                     ("ECE after T", "ECE_ts", pairs_ts, ece)):
            a_, b_ = wide.loc[f, (col, "full_ft")], wide.loc[f, (col, "lora")]
            d = (b_ - a_).values
            lo, hi = paired_t_ci(d)
            ilo, ihi = item_bootstrap_gap(pairs, fn, rng)
            print(f"  {name:12s} full {a_.mean():.4f} (sd {a_.std():.4f})  lora {b_.mean():.4f} "
                  f"(sd {b_.std():.4f})  diff {d.mean():+.4f}  seed-level [{lo:+.4f}, {hi:+.4f}] "
                  f"p={stats.ttest_rel(b_, a_).pvalue:.2g}  LoRA lower in {int((d < 0).sum())}/{len(d)}"
                  f"  item-level [{ilo:+.4f}, {ihi:+.4f}]")

    # ---- convergence ---------------------------------------------------------
    print("\n=== best (dev-selected) epoch ===")
    for f in fractions:
        be = wide.loc[f, "best_epoch"]
        d = be["lora"] - be["full_ft"]
        p = stats.ttest_rel(be["lora"], be["full_ft"]).pvalue if d.std() > 0 else float("nan")
        hits = m[m.fraction == f].groupby("method").ceiling_hit.sum()
        print(f"{f:<5g} full {sorted(be['full_ft'].astype(int))} (mean {be['full_ft'].mean():.1f})  "
              f"lora {sorted(be['lora'].astype(int))} (mean {be['lora'].mean():.1f})  "
              f"paired gap {d.mean():+.1f}  LoRA later in {int((d > 0).sum())}/{len(d)}  "
              f"p={p:.4f}  ceiling hit full {int(hits['full_ft'])}/5, lora {int(hits['lora'])}/5")

    # ---- efficiency (never pooled across GPUs) -------------------------------
    print("\n=== efficiency by GPU ===")
    ok = runs[runs.get("wall_clock_comparable", True).astype(bool)]
    print(ok.groupby(["gpu", "method"]).agg(
        runs=("run_id", "count"), trainable=("trainable_params", "max"),
        model_mb=("saved_model_mb", "mean"), peak_mb=("peak_gpu_mem_mb", "mean"),
        samples_per_s=("train_samples_per_s", "mean")).round(1).to_string())

    figures(art, args.figures, m, fractions)


def figures(art, out, m, fractions):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out, exist_ok=True)
    colors = {"full_ft": "#1f77b4", "lora": "#ff7f0e"}
    x = np.arange(len(fractions))
    xt = [f"{f:.0%}" for f in fractions]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, col, ylabel, title in (
            (axes[0], "f1", "test macro-F1", "Accuracy: no measurable gap"),
            (axes[1], "ECE", f"test ECE ({N_BINS} bins, lower is better)",
             "Calibration: gap largest at 1%")):
        for j, meth in enumerate(METHODS):
            g = m[m.method == meth].groupby("fraction")[col]
            off = (j - 0.5) * 0.18
            ax.errorbar(x + off, g.mean().values, yerr=g.std().values, fmt="o",
                        color=colors[meth], capsize=4, label=LABEL[meth])
            for i, f in enumerate(fractions):
                v = m[(m.method == meth) & (m.fraction == f)][col]
                ax.scatter(np.full(len(v), i + off + 0.05), v, s=8,
                           color=colors[meth], alpha=0.35)
        ax.set_xticks(x, xt)
        ax.set_xlabel("fraction of XNLI-bn training data")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].legend(loc="lower right")
    fig.suptitle("5 seeds per point; bars = sd, dots = individual seeds", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{out}/f1_and_ece.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(fractions), figsize=(4 * len(fractions), 3.4),
                             sharey=False)
    for ax, f in zip(np.atleast_1d(axes), fractions):
        for meth in METHODS:
            for p in sorted(glob.glob(f"{art}/curves/{meth}_frac{f:g}_seed*.json")):
                c = json.load(open(p))
                ax.plot([e["epoch"] for e in c], [e["eval_f1_macro"] for e in c],
                        color=colors[meth], alpha=0.6, lw=1.2)
            ax.plot([], [], color=colors[meth], label=LABEL[meth])
        ax.set_title(f"{f:.0%} of training data", fontsize=10)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("validation macro-F1")
    np.atleast_1d(axes)[0].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{out}/dev_curves.png", dpi=160)
    plt.close(fig)
    print(f"\nwrote {out}/f1_and_ece.png and {out}/dev_curves.png")


if __name__ == "__main__":
    main()
