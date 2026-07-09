from typing import Any, Dict, Optional

import torch
from datasets import Audio, load_dataset
from torch.utils.data import Dataset

from .feature_extractor import AudioFeatureExtractor

_TARGET_SR = 16000  # must match the extractor's expected sampling rate


class PairedDomainDataset(Dataset):
    """Pairs a labeled source dataset with an (unlabeled) target dataset so
    each `__getitem__` returns one dict Trainer can collate directly.

    If the two datasets have different lengths, the shorter one is cycled.
    """

    def __init__(self, source_dataset, target_dataset):
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.length = max(len(source_dataset), len(target_dataset))

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, Any]:  # type: ignore[override]
        x_s, labels_s = self.source_dataset[idx % len(self.source_dataset)]
        x_t = self.target_dataset[idx % len(self.target_dataset)]
        return {"x_s": x_s, "labels_s": labels_s, "x_t": x_t}


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


class EmoDBDataset(Dataset):
    """Wraps `renumics/emodb` (HF Hub speech-emotion dataset).

    Each item is (feature_embedding, emotion_label) -- the raw waveform is
    run through the given `AudioFeatureExtractor` before being returned.
    Set `with_labels=False` to use it as an unlabeled dataset (only the
    embedding is returned) -- this is how the same split doubles as the
    target domain.
    """

    def __init__(
        self,
        hf_split,
        feature_extractor: AudioFeatureExtractor,
        with_labels: bool = True,
    ):
        self.hf_split = hf_split
        self.feature_extractor = feature_extractor
        self.with_labels = with_labels

    def __len__(self) -> int:
        return len(self.hf_split)

    def __getitem__(self, idx: int):  # type: ignore[override]
        row = self.hf_split[idx]
        waveform = torch.tensor(row["audio"]["array"], dtype=torch.float32)
        x = self.feature_extractor(waveform, row["audio"]["sampling_rate"])
        if self.with_labels:
            return x, row["emotion"]
        return x


def build_emodb_datasets(
    eval_ratio: float = 0.2,
    seed: int = 42,
    feature_extractor: Optional[AudioFeatureExtractor] = None,
):
    """Loads `renumics/emodb` and reuses its (single) train split as both
    domains: a labeled source view and a label-dropped target view of the
    SAME audio. A held-out slice is kept aside for evaluation.

    Args:
        feature_extractor: an `AudioFeatureExtractor` instance (backend
            picked via its own `backend=` param, e.g. "hubert" or
            "wav2vec2"). Defaults to HuBERT if not given. One instance is
            shared across source/target/eval so the backbone is only loaded
            once.

    Returns (source_dataset, target_dataset, eval_dataset_raw).
    """
    feature_extractor = feature_extractor or AudioFeatureExtractor(
        backend="hubert", sampling_rate=_TARGET_SR, device="cpu"
    )

    full = load_dataset("renumics/emodb", split="train")
    full = full.cast_column(
        "audio", Audio(sampling_rate=_TARGET_SR)
    )  # match the extractor's expected rate
    split = full.train_test_split(test_size=eval_ratio, seed=seed)
    train_hf, eval_hf = split["train"], split["test"]

    source_dataset = EmoDBDataset(train_hf, feature_extractor, with_labels=True)
    target_dataset = EmoDBDataset(
        train_hf, feature_extractor, with_labels=False
    )  # same audio, unlabeled
    eval_dataset_raw = EmoDBDataset(eval_hf, feature_extractor, with_labels=True)
    return source_dataset, target_dataset, eval_dataset_raw
