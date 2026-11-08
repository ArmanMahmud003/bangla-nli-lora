import sys, itertools, numpy as np, pandas as pd
d = sys.argv[1]
tr, va, te = [pd.read_csv(f"{d}/bnli_{s}.csv") for s in ("train", "val", "test")]
L = ["entailment", "neutral", "contradiction"]
def mf1(y, p):
    f = []
    for c in L:
        tp = np.sum((p == c) & (y == c)); den = np.sum(p == c) + np.sum(y == c)
        f.append(2 * tp / den if den else 0.0)
    return float(np.mean(f))
a = pd.concat([tr, va, te]); s, y = a.similarity, a.label
print("similarity range", round(s.min(), 3), round(s.max(), 3), "missing", s.isna().sum())
ok = ((y == "entailment") & (s > .70)) | ((y == "neutral") & (s > .30) & (s < .75)) | ((y == "contradiction") & (s < .30))
print(f"pairs meeting the paper's keep rule: {ok.sum()} / {len(a)} ({ok.mean()*100:.1f}%)")
print(ok.groupby(y).mean().round(3).to_dict())
print(a.groupby("label").similarity.describe()[["mean", "25%", "50%", "75%"]].round(3).to_string())
def rule(x):
    return np.where(x > .70, "entailment", np.where(x < .30, "contradiction", "neutral"))
for n, x in (("val", va), ("test", te)):
    p = rule(x.similarity.values); yy = x.label.values
    print(f"paper cut-offs on {n}: acc {np.mean(p==yy):.4f} macroF1 {mf1(yy,p):.4f}")
ys, ss = tr.label.values, tr.similarity.values
grid = np.quantile(ss, np.linspace(0.02, 0.98, 49))
best = (-1, None)
for lo, hi in itertools.combinations(grid, 2):
    for perm in itertools.permutations(L):
        p = np.where(ss < lo, perm[0], np.where(ss < hi, perm[1], perm[2]))
        f = mf1(ys, p)
        if f > best[0]: best = (f, (lo, hi, perm))
f, (lo, hi, perm) = best
print(f"cut-offs fitted on train: {lo:.3f} / {hi:.3f}, low/mid/high = {perm}, train macroF1 {f:.4f}")
for n, x in (("val", va), ("test", te)):
    yy = x.label.values; p = np.where(x.similarity < lo, perm[0], np.where(x.similarity < hi, perm[1], perm[2]))
    print(f"fitted cut-offs on {n}: acc {np.mean(p==yy):.4f} macroF1 {mf1(yy,p):.4f}")
print("majority-class test acc", round(te.label.value_counts(normalize=True).max(), 4))
