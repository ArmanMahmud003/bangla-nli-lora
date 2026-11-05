"""How much do ECE-after-T verdicts depend on the arbitrary random fold assignment?
Recompute per-run ECE_after_T for 20 fold seeds, both schemes; paired t + Holm per block."""
import sys
sys.path.insert(0, sys.argv[1])
import numpy as np, pandas as pd
from scipy import stats
import analyze_v2 as A
GROUPS = {k: pd.read_csv(p).sort_values("row").premise_group.values for k, p in A.CLUSTER_FILES.items()}

def ece_T(logits, y, folds):
    out = np.zeros_like(logits)
    for te in folds:
        tr = np.setdiff1d(np.arange(len(y)), te)
        out[te] = A.softmax(logits[te] / A.fit_temperature(logits[tr], y[tr]))
    conf = out.max(1); corr = (out.argmax(1) == y).astype(float)
    b = np.searchsorted(A.EDGES, conf, side="right") - 1
    return float(np.abs(np.bincount(b, corr, A.N_BINS) - np.bincount(b, conf, A.N_BINS)).sum() / len(y))

data = {}
for ev, (fracs, path, ts) in A.EVALS.items():
    for f in fracs:
        for s in A.SEEDS:
            for m in A.METHODS:
                lg, y = A.load_npz(path(m, f, s)); data[(ev, m, f, s)] = (lg.astype(np.float64), y.astype(int), ts)
rows = []
for off in range(int(sys.argv[2])):
    for scheme in ("pair", "group"):
        val = {}
        for key, (lg, y, ts) in data.items():
            rng = np.random.default_rng(key[3] + 1000 * off)
            if scheme == "pair":
                folds = np.array_split(rng.permutation(len(y)), 5)
            else:
                g = GROUPS[ts]
                folds = [np.flatnonzero(np.isin(g, fg)) for fg in np.array_split(rng.permutation(np.unique(g)), 5)]
            val[key] = ece_T(lg, y, folds)
        for ev, (fracs, _, _) in A.EVALS.items():
            for fam, pairs in (("primary", A.PRIMARY), ("secondary", A.SECONDARY)):
                blk = []
                for f in fracs:
                    for a, b in pairs:
                        d = np.array([val[(ev, a, f, s)] - val[(ev, b, f, s)] for s in A.SEEDS])
                        blk.append(dict(off=off, scheme=scheme, eval=ev, fraction=f, a=a, b=b, diff=d.mean(),
                                        p=float(stats.ttest_1samp(d, 0).pvalue)))
                for r, p in zip(blk, A.holm([r["p"] for r in blk])):
                    r["p_holm"] = p
                rows += blk
    print("off", off, flush=True)
pd.DataFrame(rows).to_csv(sys.argv[3], index=False)
