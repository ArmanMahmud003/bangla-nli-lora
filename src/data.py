"""XNLI-bn loading, normalisation, tokenisation and the evaluation metric."""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

DATASET = "csebuetnlp/xnli_bn"
DATASET_REVISION = "refs/convert/parquet"
TOKENIZER = "csebuetnlp/banglabert"
MAX_LENGTH = 128


def load_tokenized():
    """Return {"train", "validation", "test"} tokenized datasets.

    Both sentences go through the csebuetnlp normalizer before tokenisation, as
    BanglaBERT expects. Expected sizes: 381,449 / 2,419 / 4,895.
    """
    from datasets import load_dataset
    from normalizer import normalize
    from transformers import AutoTokenizer

    dataset = load_dataset(DATASET, revision=DATASET_REVISION)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)

    def encode(batch):
        s1 = [normalize(t) for t in batch["sentence1"]]
        s2 = [normalize(t) for t in batch["sentence2"]]
        return tokenizer(s1, s2, truncation=True, padding="max_length",
                         max_length=MAX_LENGTH)

    return {split: dataset[split].map(encode, batched=True)
            for split in ("train", "validation", "test")}


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro")}
