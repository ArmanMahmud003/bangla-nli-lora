ECE after temperature scaling: repeated, premise-grouped 5-fold CV (5 Nov 2026)

Scripts (run from anywhere; arg 1 = path to the repo's src/ folder)
  repT.py              main analysis. python3 -I repT.py <repo/src> 20 0  setA
                                      python3 -I repT.py <repo/src> 20 20 setB
                       Per run: 20 random premise-grouped fold splits (fold seed = run seed + 1000*k),
                       ECE_after_T = mean over the 20. Same t-test, Holm and hierarchical bootstrap
                       (RNG 20260910, 2,000 resamples) as src/analyze_v2.py. Only ECE_after_T is recomputed.
  fold_seed_sens.py    sensitivity check: ECE_after_T with ONE fold split, 20 different split seeds,
                       pair-level and premise-grouped folds (p-values only, no bootstrap).
  sim2.py              similarity-only baseline on BNLI (cut-offs fitted on train). Arg: folder with
                       bnli_train/val/test.csv from src/make_bnli_split.py.

Results
  setA_*  = splits k = 0..19   -> numbers used in the thesis
  setB_*  = splits k = 20..39  -> independent stability check
  Set A and set B agree on every robust/not-robust verdict except
  LoRA - full FT, BNLI -> BNLI (set A robust, CI upper end -0.0002; set B not robust).
  fold_seed_sensitivity_20splits.csv = one-split verdicts across 20 split seeds.
