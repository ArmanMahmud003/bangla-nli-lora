"""Four-method analysis (full FT, LoRA, DoRA, IA3) on XNLI-bn and BNLI, both directions.

    python src/analyze_v2.py                 # writes results/v2/*.csv and results/v2/stats_v2.txt
    N_BOOT=200 python src/analyze_v2.py      # quick run

Which runs are used
  XNLI-bn-trained, all on an A100: full FT and LoRA from results/preds, except the 14
  runs that first ran on a T4, which are replaced by their A100 re-runs in
  results/peft/preds; DoRA and IA3 from results/peft/preds. Their BNLI-test scores
  (cross-dataset) come from results/peft/preds (*_xtest) and results/xbnli/preds.
  BNLI-trained: results/bnli/preds (*_test = BNLI test, *_xtest = XNLI-bn test).

Statistics (decided before this script was run)
  For each comparison A - B (B = full FT in the primary family; B = LoRA for DoRA and
  IA3 in the secondary family) and each metric:
    seed-level   paired t over the five seeds (95% CI, two-sided p)
    hierarchical bootstrap: resample seeds, then test items inside each drawn seed
                 test items are resampled by premise group on both test sets, because
                 pairs share premises (XNLI-bn test: 4,895 pairs, 1,651 premises;
                 BNLI test: 2,252 pairs, 663 premises)
  Holm correction within each family (evaluation set x metric x family).
  A difference is ROBUST only if the hierarchical 95% CI excludes 0 AND the
  Holm-adjusted seed-level p < 0.05.
  Macro-F1 equivalence: TOST at +/-0.01 on the seed-level differences (margin chosen
  after the original runs, not pre-registered), Holm-adjusted within the same family.
  Temperature scaling: only test logits exist, so T is fitted by 5-fold CV on the test
  set (cv seed = run seed), as in the original analysis.
"""
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
OUT = os.environ.get("OUT_DIR") or os.path.join(ROOT, "v2")
N_BOOT = int(os.environ.get("N_BOOT", 2000))
RNG_SEED = 20260910
SEEDS = [7, 13, 21, 33, 42]
FRACTIONS = [0.01, 0.05, 0.1]
METHODS = ["full_ft", "lora", "dora", "ia3"]
T4_SEEDS = {0.01: [7, 13, 21, 33], 0.05: [7, 13, 42]}   # original runs on a T4 (run manifests)
N_BINS = 15
TOST_MARGIN = 0.01
PRIMARY = [("lora", "full_ft"), ("dora", "full_ft"), ("ia3", "full_ft")]
SECONDARY = [("dora", "lora"), ("ia3", "lora")]


# ---------------------------------------------------------------- metrics
def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def nll_mean(p, y):
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1.0)).mean())


def fit_temperature(logits, y):
    return minimize_scalar(lambda t: nll_mean(softmax(logits / t), y),
                           bounds=(0.05, 10.0), method="bounded").x


def cv_temperature(logits, y, seed, k=5):
    folds = np.array_split(np.random.default_rng(seed).permutation(len(y)), k)
    out, temps = np.zeros_like(logits), []
    for i, test in enumerate(folds):
        train = np.concatenate([f for j, f in enumerate(folds) if j != i])
        t = fit_temperature(logits[train], y[train])
        temps.append(t)
        out[test] = softmax(logits[test] / t)
    return out, float(np.mean(temps))


EDGES = np.linspace(1.0 / 3, 1.0, N_BINS + 1)
EDGES[0], EDGES[-1] = 0.0, 1.0 + 1e-9


class Run:
    """Per-item quantities, so every bootstrap resample is a cheap indexing operation."""

    def __init__(self, logits, y, seed):
        logits = logits.astype(np.float64)
        p = softmax(logits)
        self.y = y.astype(int)
        self.pred = p.argmax(1)
        self.correct = (self.pred == self.y).astype(float)
        self.conf = p.max(1)
        self.bin = np.searchsorted(EDGES, self.conf, side="right") - 1
        self.brier = ((p - np.eye(3)[self.y]) ** 2).sum(1)
        self.nll = -np.log(np.clip(p[np.arange(len(self.y)), self.y], 1e-12, 1.0))
        pt, self.temperature = cv_temperature(logits, self.y, seed)
        self.conf_t = pt.max(1)
        self.bin_t = np.searchsorted(EDGES, self.conf_t, side="right") - 1

    def metric(self, name, idx=None):
        idx = slice(None) if idx is None else idx
        if name == "macro_f1":
            y, pr = self.y[idx], self.pred[idx]
            f1 = []
            for c in range(3):
                tp = np.sum((pr == c) & (y == c))
                den = 2 * tp + np.sum((pr == c) & (y != c)) + np.sum((pr != c) & (y == c))
                f1.append(2 * tp / den if den else 0.0)
            return float(np.mean(f1))
        if name in ("ECE", "ECE_after_T"):
            conf = self.conf[idx] if name == "ECE" else self.conf_t[idx]
            b = self.bin[idx] if name == "ECE" else self.bin_t[idx]
            corr = self.correct[idx]
            s_corr = np.bincount(b, weights=corr, minlength=N_BINS)
            s_conf = np.bincount(b, weights=conf, minlength=N_BINS)
            return float(np.abs(s_corr - s_conf).sum() / len(corr))
        if name == "Brier":
            return float(self.brier[idx].mean())
        if name == "NLL":
            return float(self.nll[idx].mean())
        raise ValueError(name)


METRICS = ["macro_f1", "ECE", "Brier", "NLL", "ECE_after_T"]


# ---------------------------------------------------------------- loading
def load_npz(path):
    d = np.load(path)
    return d["logits"], d["labels"]


def xnli_path(method, frac, seed, kind):
    """kind 'test' = XNLI-bn test, 'xtest' = BNLI test. All A100."""
    rid = f"{method}_frac{frac}_seed{seed}"
    if method in ("dora", "ia3") or seed in T4_SEEDS.get(frac, []):
        return f"{ROOT}/peft/preds/{rid}_{kind}.npz"
    return f"{ROOT}/preds/{rid}_test.npz" if kind == "test" else f"{ROOT}/xbnli/preds/{rid}_xtest.npz"


def bnli_path(method, seed, kind):
    return f"{ROOT}/bnli/preds/{method}_frac1.0_seed{seed}_{kind}.npz"


EVALS = {   # name: (fractions, path function, test set whose premises are resampled)
    # third field = which test set's premise groups to resample in the bootstrap
    "xnli_to_xnli": (FRACTIONS, lambda m, f, s: xnli_path(m, f, s, "test"), "xnli"),
    "xnli_to_bnli": (FRACTIONS, lambda m, f, s: xnli_path(m, f, s, "xtest"), "bnli"),
    "bnli_to_bnli": ([1.0], lambda m, f, s: bnli_path(m, s, "test"), "bnli"),
    "bnli_to_xnli": ([1.0], lambda m, f, s: bnli_path(m, s, "xtest"), "xnli"),
}


def load_all():
    runs = {}
    for ev, (fracs, path, _) in EVALS.items():
        for f in fracs:
            for s in SEEDS:
                ys = None
                for m in METHODS:
                    logits, y = load_npz(path(m, f, s))
                    if ys is None:
                        ys = y
                    assert np.array_equal(y, ys), f"labels differ: {ev} {m} {f} {s}"
                    runs[(ev, m, f, s)] = Run(logits, y, s)
    return runs


def premise_clusters(path):
    """Index arrays, one per premise group, in the row order of the saved predictions."""
    g = pd.read_csv(path).sort_values("row")
    return [np.flatnonzero(g.premise_group.values == k) for k in np.unique(g.premise_group.values)]


CLUSTER_FILES = {"bnli": f"{ROOT}/bnli/test_premise_groups.csv",
                 "xnli": f"{ROOT}/xnli_test_premise_groups.csv"}


# ---------------------------------------------------------------- statistics
def holm(p):
    p = np.asarray(p, float)
    order, adj, running = np.argsort(p), np.empty(len(p)), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def compare(runs, ev, f, a, b, metric, clusters, rng):
    pairs = {s: (runs[(ev, a, f, s)], runs[(ev, b, f, s)]) for s in SEEDS}
    d = np.array([ra.metric(metric) - rb.metric(metric) for ra, rb in pairs.values()])
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    tcrit = stats.t.ppf(0.975, n - 1)
    p_t = float(stats.ttest_1samp(d, 0.0).pvalue) if se > 0 else 0.0
    n_items = len(next(iter(pairs.values()))[0].y)
    boot = np.empty(N_BOOT)
    for i in range(N_BOOT):
        diffs = []
        for s in rng.choice(SEEDS, n, replace=True):
            if clusters is None:
                idx = rng.integers(0, n_items, n_items)
            else:
                idx = np.concatenate([clusters[k] for k in rng.integers(0, len(clusters), len(clusters))])
            ra, rb = pairs[s]
            diffs.append(ra.metric(metric, idx) - rb.metric(metric, idx))
        boot[i] = np.mean(diffs)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    row = dict(eval=ev, fraction=f, a=a, b=b, metric=metric,
               mean_a=np.mean([ra.metric(metric) for ra, _ in pairs.values()]),
               mean_b=np.mean([rb.metric(metric) for _, rb in pairs.values()]),
               diff=d.mean(), t_lo=d.mean() - tcrit * se, t_hi=d.mean() + tcrit * se, p_t=p_t,
               hier_lo=lo, hier_hi=hi, a_lower=int((d < 0).sum()),
               per_seed=" ".join(f"{x:+.4f}" for x in d))
    if metric == "macro_f1":
        row["p_tost"] = float(max(1 - stats.t.cdf((d.mean() + TOST_MARGIN) / se, n - 1),
                                  stats.t.cdf((d.mean() - TOST_MARGIN) / se, n - 1))) if se > 0 else 0.0
    return row


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)
    runs = load_all()
    clusters = {k: premise_clusters(p) for k, p in CLUSTER_FILES.items()}

    # per-run table
    per_run = pd.DataFrame([dict(eval=ev, method=m, fraction=f, seed=s,
                                 **{k: r.metric(k) for k in METRICS}, temperature=r.temperature)
                            for (ev, m, f, s), r in runs.items()])
    per_run.to_csv(f"{OUT}/per_run_v2.csv", index=False)
    summary = (per_run.groupby(["eval", "fraction", "method"])[METRICS + ["temperature"]]
               .agg(["mean", "std"]).round(4))
    summary.to_csv(f"{OUT}/summary_v2.csv")

    # comparisons
    rows = []
    for ev, (fracs, _, test_set) in EVALS.items():
        for fam, pairs in (("primary", PRIMARY), ("secondary", SECONDARY)):
            for metric in METRICS:
                block = [dict(compare(runs, ev, f, a, b, metric, clusters[test_set], rng), family=fam)
                         for f in fracs for a, b in pairs]
                adj = holm([r["p_t"] for r in block])
                adj_tost = holm([r["p_tost"] for r in block]) if metric == "macro_f1" else [np.nan] * len(block)
                for r, p, pt in zip(block, adj, adj_tost):
                    r["p_holm"] = p
                    r["p_tost_holm"] = pt
                    r["robust"] = bool((r["hier_lo"] > 0 or r["hier_hi"] < 0) and p < 0.05)
                    r["equivalent_0.01"] = bool(pt < 0.05) if metric == "macro_f1" else None
                rows.extend(block)
                print(f"done {ev:13s} {fam:9s} {metric}", flush=True)
    comp = pd.DataFrame(rows)
    comp.to_csv(f"{OUT}/comparisons_v2.csv", index=False)

    # cross-dataset drop (in-domain minus cross-dataset macro-F1)
    w = per_run.pivot_table(index=["method", "fraction", "seed"], columns="eval", values="macro_f1").reset_index()
    w["drop"] = np.where(w.fraction == 1.0, w.bnli_to_bnli - w.bnli_to_xnli, w.xnli_to_xnli - w.xnli_to_bnli)
    drop = w.groupby(["fraction", "method"])["drop"].agg(["mean", "std"]).round(4)
    drop.to_csv(f"{OUT}/cross_dataset_drop_v2.csv")

    # hardware: original T4 runs vs their A100 re-runs (same method, fraction, seed)
    hw = []
    for f, seeds in T4_SEEDS.items():
        for s in seeds:
            for m in ("full_ft", "lora"):
                t4 = Run(*load_npz(f"{ROOT}/preds/{m}_frac{f}_seed{s}_test.npz"), s)
                a1 = runs[("xnli_to_xnli", m, f, s)]
                hw.append(dict(method=m, fraction=f, seed=s,
                               f1_t4=t4.metric("macro_f1"), f1_a100=a1.metric("macro_f1"),
                               ece_t4=t4.metric("ECE"), ece_a100=a1.metric("ECE")))
    hw = pd.DataFrame(hw)
    hw["f1_diff"], hw["ece_diff"] = hw.f1_a100 - hw.f1_t4, hw.ece_a100 - hw.ece_t4
    hw.to_csv(f"{OUT}/hardware_t4_vs_a100.csv", index=False)

    with open(f"{OUT}/stats_v2.txt", "w") as fh:
        pd.set_option("display.width", 250)
        fh.write(f"N_BOOT={N_BOOT}  RNG_SEED={RNG_SEED}  TOST margin={TOST_MARGIN}\n\n== summary ==\n")
        fh.write(summary.to_string() + "\n\n== comparisons ==\n")
        cols = ["eval", "family", "fraction", "a", "b", "metric", "mean_a", "mean_b", "diff", "t_lo", "t_hi",
                "hier_lo", "hier_hi", "p_t", "p_holm", "robust", "p_tost", "p_tost_holm", "equivalent_0.01", "a_lower"]
        fh.write(comp[cols].round(4).to_string(index=False) + "\n\n== cross-dataset drop ==\n")
        fh.write(drop.to_string() + "\n\n== hardware (A100 minus T4, same run) ==\n")
        fh.write(hw.round(4).to_string(index=False) + "\n")
    print("written to", OUT)


if __name__ == "__main__":
    main()
