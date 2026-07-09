from typing import Any, Dict

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch.utils.data import Dataset
from transformers import EvalPrediction


class ClassificationEvalDataset(Dataset):
    """Wraps a plain (x, labels) dataset for evaluation.

    Evaluation only needs the classifier head (source-domain forward pass),
    not the target-domain / SSL / OT machinery, so this deliberately does
    NOT require an `x_t` field.
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:  # type: ignore[override]
        x, labels = self.dataset[idx]
        return {"x_s": x, "labels_s": labels}


def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    """Weighted accuracy (standard accuracy), unweighted accuracy (macro/
    balanced accuracy -- average per-class recall), and F1 (macro + weighted).
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "weighted_accuracy": accuracy_score(labels, preds),
        "unweighted_accuracy": balanced_accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
    }
