"""ECE after temperature scaling with REPEATED premise-grouped 5-fold CV.
Per run: R random premise-grouped fold splits (fold seed = run seed + 1000*(rep+REP0));
ECE_after_T = mean over the R splits. Bootstrap: same hierarchical scheme as analyze_v2,
each resample evaluates the R-split mean on the resampled items.
usage: python3 -I src/ece_after_t_repeated_cv.py src <R> <REP0> <out-prefix>  (see results/v2/ece_after_T_repeated/README.md)"""
import sys
sys.path.insert(0, sys.argv[1])
import numpy as np, pandas as pd
import analyze_v2 as A
R, REP0, OUT = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
GROUPS = {k: pd.read_csv(p).sort_values("row").premise_group.values for k, p in A.CLUSTER_FILES.items()}
NB = A.N_BINS

class RunT:
    def __init__(self, logits, y, seed, g):
        logits = logits.astype(np.float64); y = y.astype(int); n = len(y)
        conf = np.empty((R, n)); corr = np.empty((R, n)); temps = []
        uniq = np.unique(g)
        for r in range(R):
            rng = np.random.default_rng(seed + 1000 * (r + REP0))
            out = np.zeros_like(logits)
            for fg in np.array_split(rng.permutation(uniq), 5):
                te = np.isin(g, fg)
                t = A.fit_temperature(logits[~te], y[~te]); temps.append(t)
                out[te] = A.softmax(logits[te] / t)
            conf[r] = out.max(1); corr[r] = (out.argmax(1) == y)
        self.conf, self.corr, self.temperature = conf, corr, float(np.mean(temps))
        self.bin = np.searchsorted(A.EDGES, conf, side="right") - 1 + NB * np.arange(R)[:, None]
        self.y = y
    def metric(self, name, idx=None):
        assert name == "ECE_after_T"
        idx = slice(None) if idx is None else idx
        b = self.bin[:, idx].ravel(); m = self.bin[:, idx].shape[1]
        sc = np.bincount(b, self.corr[:, idx].ravel(), NB * R); sf = np.bincount(b, self.conf[:, idx].ravel(), NB * R)
        return float(np.abs(sc - sf).reshape(R, NB).sum(1).mean() / m)

runs = {}
for ev, (fracs, path, ts) in A.EVALS.items():
    for f in fracs:
        for s in A.SEEDS:
            for m in A.METHODS:
                lg, y = A.load_npz(path(m, f, s)); runs[(ev, m, f, s)] = RunT(lg, y, s, GROUPS[ts])
print("runs built", flush=True)
pd.DataFrame([dict(eval=ev, method=m, fraction=f, seed=s, ECE_after_T=r.metric("ECE_after_T"), T=r.temperature)
              for (ev, m, f, s), r in runs.items()]).to_csv(OUT + "_per_run.csv", index=False)
rng = np.random.default_rng(A.RNG_SEED)
clusters = {k: A.premise_clusters(p) for k, p in A.CLUSTER_FILES.items()}
rows = []
for ev, (fracs, _, ts) in A.EVALS.items():
    for fam, pairs in (("primary", A.PRIMARY), ("secondary", A.SECONDARY)):
        blk = [dict(A.compare(runs, ev, f, a, b, "ECE_after_T", clusters[ts], rng), family=fam) for f in fracs for a, b in pairs]
        for r, p in zip(blk, A.holm([r["p_t"] for r in blk])):
            r["p_holm"] = p; r["robust"] = bool((r["hier_lo"] > 0 or r["hier_hi"] < 0) and p < 0.05)
        rows += blk
        print("done", ev, fam, flush=True)
pd.DataFrame(rows).to_csv(OUT + "_comp.csv", index=False)
print("ALL DONE")
