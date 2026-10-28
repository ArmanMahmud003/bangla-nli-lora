# Results folder map

Which files hold the final thesis results, which are superseded, and what the file names mean.
**Folders and files were not renamed** so that every path used by the code still works.

## Final results (cited in the thesis)

| Path | What it is |
|---|---|
| `v2/` | **Final four-method analysis** (full FT, LoRA, DoRA, IA3), all XNLI-bn runs on an A100. Written by `src/analyze_v2.py` and `src/analyze_v2_extra.py`. Start with `v2/stats_v2.txt` and `v2/comparisons_v2.csv`. |
| `v2/hardware_t4_vs_a100.csv` | Same run on T4 vs A100: how much the GPU alone moves macro-F1 and ECE. |
| `peft/` | DoRA and IA3 runs on XNLI-bn (3 fractions x 5 seeds), **plus the 14 full-FT/LoRA A100 re-runs** that replace the original T4 runs. |
| `peft/ia3_lrcheck/` | IA3 learning-rate check (validation only, 5% data, seed 0). `summary.csv` is the one-line-per-rate table; `results.csv` is the full run log; `ia3_lr_choice.json` is the chosen rate. |
| `bnli/` | All four methods trained on BNLI (5 seeds). `split_sha256.txt` fixes the split files; `test_premise_groups.csv` is used to resample premises in the bootstrap. |
| `xnli_test_premise_groups.csv` | Premise group of each XNLI-bn test pair (4,895 pairs, 1,651 premises), in the row order of every saved XNLI-bn test prediction; its `label` column was checked against all 102 such files. Used to resample premises in the bootstrap on the XNLI-bn test set. |
| `xbnli/` | XNLI-bn-trained models scored on the BNLI test set for the runs that did not score it during training: 8 full-FT A100 re-runs (`runs/`, `preds/`) and 8 saved LoRA adapters scored by inference (`lora_adapter_xtest.csv`). |
| `runs/`, `curves/`, `preds/` (top level) | The original 30 full-FT/LoRA XNLI-bn runs. **Only the 16 A100 runs are used by the final analysis**; the 14 runs whose `gpu` is `Tesla T4` are replaced by their A100 re-runs in `peft/`. |

## Superseded (kept for the record; do not cite)

| Path | Why it is superseded |
|---|---|
| `results.csv`, `stats.txt` | Two-method analysis of the original 30 runs on **mixed T4/A100 hardware**. Its "equivalence" wording and gap values are replaced by `v2/`. |
| `robustness/*_v2.csv` (and `../analysis/robustness_v2.ipynb`) | Two-method (LoRA vs full FT) calibration CIs from 20 Oct 2026, on mixed hardware. Replaced by the four-method analysis in `v2/`. The "v2" here is **not** the same as `v2/`. |
| `superseded_runs.csv` | Six runs from an earlier pass (T4 / RTX 5060) replaced by A100 re-runs **before** the 30-run grid. Not the same as the 14 T4 runs above. |
| `../src/analyze.py` | Script behind `results.csv` / `stats.txt`. |
| `../figures/dev_curves.png`, `../figures/f1_and_ece.png` | Two-method figures from the original analysis. Final figures are `../figures/v2_*.png`. |

## File names

Per-run files are named `{method}_frac{fraction}_seed{seed}`.

- `method`: `full_ft`, `lora`, `dora`, `ia3`; `ia3lr{rate}` only in the IA3 learning-rate check.
- `fraction`: share of the XNLI-bn training set (`0.01`, `0.05`, `0.1`); `1.0` for BNLI runs (whole BNLI training split).
- `runs/*.json`: one manifest per run (settings, scores, hardware, library versions).
- `curves/*.json`: validation macro-F1 after each epoch.
- `preds/*_test.npz`: test-set logits on the **training dataset's own** test split.
- `preds/*_xtest.npz`: test-set logits on the **other** dataset (cross-dataset). Its meaning depends on where the run lives:
  - XNLI-bn-trained runs (`peft/`, `xbnli/`): `xtest` = **BNLI test**.
  - BNLI-trained runs (`bnli/`): `xtest` = **XNLI-bn test**.
