# BNLI: how much does the similarity score alone predict the label?

`BNLI_Dataset.xlsx` has a `Similarity` column for every pair. The BNLI paper filters pairs with a
fixed similarity rule (keep entailment above 0.70, neutral between 0.30 and 0.75, contradiction below
0.30), so labels may be predictable from similarity alone, without reading the sentences.

```
python3 -I src/make_bnli_split.py BNLI_Dataset.xlsx out 100
python3 -I src/bnli_similarity_baseline.py out > results/bnli/similarity_baseline/output.txt
```

Result (`output.txt`), BNLI test split:

| | Accuracy | Macro-F1 |
|---|---|---|
| Majority class | 0.336 | – |
| Two cut-offs on similarity, fitted on train | 0.592 | 0.596 |
| The paper's own cut-offs (0.30 / 0.70) | 0.580 | 0.587 |

Only 60.6% of the cleaned pairs satisfy the paper's keep rule, and the column takes values below 0
(minimum −0.241), whereas the paper describes scores in [0, 1]. So either the column is not the
score the paper filtered on, or the rule was not applied as described.

This shows the shortcut **exists in the data**. It does not show whether the models use it.
