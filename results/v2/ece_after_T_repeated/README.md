# ECE after temperature scaling: repeated, premise-grouped cross-validation

Replaces the `ECE_after_T` numbers in `results/v2/comparisons_v2.csv` and `per_run_v2.csv`.
All other metrics are unchanged.

## Why

`analyze_v2.py` fits the temperature by 5-fold CV on the test set, with **one** random split into
folds per run, built from **individual pairs**. Two problems:

1. Pairs that share a premise could land in different folds (the bootstrap, by contrast, resamples
   by premise group).
2. ECE after scaling depends on which random fold split is used. With 20 different single splits
   (`single_split_sensitivity_20seeds.csv`), the number of robust ECE-after-T differences ranged
   from 1 to 5, and IA3 − full FT at 1% (XNLI-bn test) was significant in 80% of splits although
   `comparisons_v2.csv` reports it as not robust (p = 0.060). (Counts hold each bootstrap CI fixed
   at its original value; p-values are exact.)

Point 1 turned out not to matter (premise-grouped folds changed per-run ECE after T by +0.0002 on
average). Point 2 does.

## What was done

Per run: **20 random premise-grouped 5-fold splits** (fold seed = run seed + 1000·k); ECE after T
is the mean over the 20. Same seed-level t-test, Holm correction and hierarchical bootstrap
(2,000 resamples, RNG 20260910) as `analyze_v2.py`; each bootstrap resample evaluates the 20-split
mean on the resampled items.

```
python3 -I src/ece_after_t_repeated_cv.py src 20 0  results/v2/ece_after_T_repeated/setA   # k = 0..19
python3 -I src/ece_after_t_repeated_cv.py src 20 20 results/v2/ece_after_T_repeated/setB   # k = 20..39
python3 -I src/ece_after_t_fold_sensitivity.py src 20 results/v2/ece_after_T_repeated/single_split_sensitivity_20seeds.csv
```

- `setA_*`: the numbers used in the thesis.
- `setB_*`: an independent set of 20 splits, as a stability check.

## Verdict changes (set A vs `comparisons_v2.csv`)

| Comparison | Old | New (set A) | Set B |
|---|---|---|---|
| IA3 − full FT, XNLI-bn 1% → XNLI-bn | not robust (0.060) | robust (0.039) | robust |
| IA3 − full FT, BNLI → XNLI-bn | robust (0.047) | not robust (0.061) | not robust |
| IA3 − LoRA, XNLI-bn 10% → BNLI | robust (0.049) | not robust (0.059) | not robust |
| LoRA − full FT, BNLI → BNLI | not robust | robust, CI upper end −0.0002 | not robust |

Sets A and B agree on every other verdict. The last row is unstable and must be reported as such.
