# Pilot runs before the main experiments (August 2026)

These files are the surviving records of the runs made **before** the fixed 8-epoch, no-early-stopping
design. None of them is used in any comparison in the thesis. They are kept because the thesis
justifies the fixed budget by what happened here. Originals are in Google Drive,
`MyDrive/thesis_bangla_nli/`.

| File | What it is |
|---|---|
| `results_log.csv` | First pass, LoRA vs full FT on XNLI-bn, seeds 42 and 7, fractions 1%–100%. Validation scores only. |
| `pipeline_first_pass_aug2026.ipynb` | Notebook that produced `results_log.csv` (outputs included). |
| `step10_retrain_matched_budget_pilot.ipynb` | Next notebook: matched 8-epoch ceiling with early stopping; contains the one-pair pilot (1%, seed 42). |
| `trainer_state_lora_frac1.0_seed7.json` | Hugging Face trainer state of one first-pass run (100% data, LoRA, seed 7), per-epoch validation scores. |

## What the records show

**First pass (`results_log.csv`).** The two methods had *different* epoch limits, with early stopping
(patience 2) on top: full fine-tuning 3 epochs (≤ 10% data) or 2 epochs (≥ 25%); LoRA 8 or 4.
In 9 of the 12 full fine-tuning runs, the best validation epoch was the last epoch allowed
(`best_epoch == epochs`), so full fine-tuning was cut off by its limit, not by early stopping.
Example, 1%, seed 42 (from `all_curves.json` in Drive): validation macro-F1 0.674 → 0.695 → 0.706
over its 3 epochs, still rising. LoRA hit its limit in 5 of 12 runs.

**Matched-budget pilot (Step 10 notebook, cell 5).** Both methods got an 8-epoch ceiling with
patience 2. Full FT (1%, seed 42) was stopped by early stopping after 7 evaluations, best epoch 5.
For LoRA, early stopping was **silently disabled** (the log says
`early stopping required metric_for_best_model, but did not find eval_f1_macro`), so it ran all 8
epochs, best epoch 6.

## Not found

The private-repo README says that in a pilot "patience-2 cut full fine-tuning off at epoch 4 with its
best at epoch 2, while LoRA ran to convergence; removing early stopping recovered +0.0161 macro-F1
for full FT alone." No run log, notebook output or results file found in Drive (5–6 Nov 2026)
supports that specific statement. Do not cite it.
