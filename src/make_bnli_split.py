"""Clean BNLI (option B) and build a premise-grouped 80/10/10 split.

Usage: python3 -I make_bnli_split.py <BNLI_Dataset.xlsx> <out_dir> [split_seed]

Cleaning (option B):
  1. drop rows with a missing premise/hypothesis/label
  2. drop rows whose label is not entailment/neutral/contradiction
  3. keep one copy of each identical (premise, hypothesis, label) pair
  4. drop every pair that appears with conflicting labels
Split: groups = premises (whitespace- and punctuation-insensitive key), shuffled
with split_seed, then filled test -> val -> train to hit 10% / 10% / 80% of rows.
"""
import csv
import collections
import os
import random
import re
import sys
import unicodedata

import openpyxl

LABELS = ("entailment", "neutral", "contradiction")
SRC, OUT = sys.argv[1], sys.argv[2]
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 100
os.makedirs(OUT, exist_ok=True)


def ws_norm(s):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(s))).strip()


def strict_key(s):
    # whitespace + punctuation insensitive; only ever MERGES premise groups
    t = ws_norm(s)
    return "".join(ch for ch in t if not unicodedata.category(ch).startswith(("P", "Z", "S")))


rows = list(openpyxl.load_workbook(SRC, read_only=True).worksheets[0].iter_rows(values_only=True))[1:]
stats = collections.OrderedDict()
stats["rows_in_sheet"] = len(rows)

step1 = [r for r in rows if r[0] is not None and r[1] is not None and r[2] is not None]
stats["after_drop_missing"] = len(step1)
step2 = [(ws_norm(r[0]), ws_norm(r[1]), ws_norm(r[2]), r[3]) for r in step1 if ws_norm(r[2]) in LABELS]
stats["after_drop_invalid_label"] = len(step2)

labels_per_pair = collections.defaultdict(set)
for p, h, y, s in step2:
    labels_per_pair[(p, h)].add(y)
conflicts = {k for k, v in labels_per_pair.items() if len(v) > 1}
stats["conflicting_pairs_dropped"] = len(conflicts)

seen, clean = set(), []
for p, h, y, s in step2:
    if (p, h) in conflicts or (p, h) in seen:
        continue
    seen.add((p, h))
    clean.append((p, h, y, s))
stats["clean_rows"] = len(clean)

groups = collections.defaultdict(list)
for row in clean:
    groups[strict_key(row[0])].append(row)
plain_groups = len({r[0] for r in clean})
stats["distinct_premises_whitespace_norm"] = plain_groups
stats["premise_groups_strict_key"] = len(groups)

keys = sorted(groups)  # deterministic order before shuffling
random.Random(SEED).shuffle(keys)
n = len(clean)
targets = {"test": round(0.10 * n), "val": round(0.10 * n)}
split_rows = {"test": [], "val": [], "train": []}
cur = "test"
for k in keys:
    if cur == "test" and len(split_rows["test"]) >= targets["test"]:
        cur = "val"
    if cur == "val" and len(split_rows["val"]) >= targets["val"]:
        cur = "train"
    split_rows[cur].extend(groups[k])

group_of = {}
for name, rs in split_rows.items():
    for r in rs:
        group_of.setdefault(strict_key(r[0]), set()).add(name)
leaks = [g for g, s in group_of.items() if len(s) > 1]

for name in ("train", "val", "test"):
    with open(os.path.join(OUT, f"bnli_{name}.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["premise", "hypothesis", "label", "similarity"])
        w.writerows(split_rows[name])

print("SPLIT_SEED", SEED)
for k, v in stats.items():
    print(f"{k}: {v}")
print("premise_groups_overlapping_splits:", len(leaks))
for name in ("train", "val", "test"):
    rs = split_rows[name]
    c = collections.Counter(r[2] for r in rs)
    g = len({strict_key(r[0]) for r in rs})
    print(f"{name}: rows={len(rs)} ({100*len(rs)/n:.1f}%) premises={g} "
          f"ent={c['entailment']} neu={c['neutral']} con={c['contradiction']}")
