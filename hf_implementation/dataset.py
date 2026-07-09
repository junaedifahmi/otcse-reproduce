"""
Dataset wrappers for OTCSE.

Contains the two dataset classes shared by training/evaluation, plus an
EmoDB loader that uses `renumics/emodb` (from the HF Hub) as BOTH the
labeled source domain and the unlabeled target domain -- i.e. the same
535 utterances, just with labels dropped for the target view.

Audio is turned into features via `feature_extractor.AudioFeatureExtractor`
(HuBERT by default, swappable to Wav2Vec2 etc.) before it ever reaches the
model: each waveform is passed through the backbone and mean/max-pooled
over time into a single fixed-size embedding.
"""

import csv
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import soundfile as sf
import torch
from datasets import Audio, load_dataset
from torch.utils.data import Dataset

try:
    from .feature_extractor import AudioFeatureExtractor
except ImportError:  # running as a plain script, not `python -m package.module`
    from feature_extractor import AudioFeatureExtractor

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


class AudioClassificationDataset(Dataset):
    """Wraps any HF Hub audio-classification dataset (one with a standard
    `Audio` feature column + an integer label column).

    Each item is (feature_embedding, label) -- the raw waveform is run
    through the given `AudioFeatureExtractor` before being returned. Set
    `with_labels=False` to use it as an unlabeled dataset (only the
    embedding is returned) -- this is how one split can double as an
    unlabeled target domain.
    """

    def __init__(
        self,
        hf_split,
        feature_extractor: AudioFeatureExtractor,
        audio_column: str = "audio",
        label_column: str = "label",
        with_labels: bool = True,
    ):
        self.hf_split = hf_split
        self.feature_extractor = feature_extractor
        self.audio_column = audio_column
        self.label_column = label_column
        self.with_labels = with_labels

    def __len__(self) -> int:
        return len(self.hf_split)

    def __getitem__(self, idx: int):  # type: ignore[override]
        row = self.hf_split[idx]
        audio = row[self.audio_column]
        waveform = torch.tensor(audio["array"], dtype=torch.float32)
        x = self.feature_extractor(waveform, audio["sampling_rate"])
        if self.with_labels:
            return x, row[self.label_column]
        return x


def build_audio_datasets(
    dataset_name: str,
    split: str = "train",
    audio_column: str = "audio",
    label_column: str = "label",
    eval_ratio: float = 0.2,
    seed: int = 42,
    feature_extractor: Optional[AudioFeatureExtractor] = None,
    **load_dataset_kwargs,
):
    """Loads an audio-classification dataset -- from the HF Hub OR purely
    local data -- and reuses one split as both domains: a labeled source
    view and a label-dropped target view of the SAME audio. A held-out
    slice is kept aside for evaluation.

    Args:
        dataset_name: HF Hub repo id (e.g. "renumics/emodb"), OR a loader
            name for local data (e.g. "audiofolder", "csv") -- anything
            `datasets.load_dataset` accepts as its first argument.
        split: which split to load and then carve into train/eval, e.g.
            "train". If your dataset already has a separate split that
            represents a genuine domain shift (different speakers/
            conditions), load that as the target domain manually instead
            of relying on this function's source==target reuse.
        audio_column / label_column: column names in the dataset.
        feature_extractor: an `AudioFeatureExtractor` instance (backend
            picked via its own `backend=` param, e.g. "hubert" or
            "wav2vec2"). Defaults to HuBERT if not given. One instance is
            shared across source/target/eval so the backbone is only loaded
            once.
        **load_dataset_kwargs: passed straight through to `load_dataset`,
            e.g. `data_dir="path/to/local/folder"` for a folder of audio
            files organized by class, or `data_files=...` for a CSV/JSON
            manifest. See:
            https://huggingface.co/docs/datasets/en/audio_dataset

    Returns (source_dataset, target_dataset, eval_dataset_raw).
    """
    feature_extractor = feature_extractor or AudioFeatureExtractor(
        backend="hubert", sampling_rate=_TARGET_SR
    )

    full = load_dataset(dataset_name, split=split, **load_dataset_kwargs)
    full = full.cast_column(
        audio_column, Audio(sampling_rate=_TARGET_SR)
    )  # match the extractor's expected rate
    split_data = full.train_test_split(test_size=eval_ratio, seed=seed)
    train_hf, eval_hf = split_data["train"], split_data["test"]

    source_dataset = AudioClassificationDataset(
        train_hf, feature_extractor, audio_column, label_column, with_labels=True
    )
    target_dataset = AudioClassificationDataset(
        train_hf, feature_extractor, audio_column, label_column, with_labels=False
    )  # same audio, unlabeled
    eval_dataset_raw = AudioClassificationDataset(
        eval_hf, feature_extractor, audio_column, label_column, with_labels=True
    )
    return source_dataset, target_dataset, eval_dataset_raw


class LocalAudioDataset(Dataset):
    """Wraps a plain list of (filepath, label) pairs -- for data that isn't
    on the HF Hub and doesn't fit a `datasets`-library loader either.

    Each item is (feature_embedding, label), same as `AudioClassificationDataset`.
    Set `with_labels=False` for an unlabeled view.
    """

    def __init__(
        self,
        manifest: Sequence[Tuple[str, int]],
        feature_extractor: AudioFeatureExtractor,
        with_labels: bool = True,
    ):
        self.manifest = manifest
        self.feature_extractor = feature_extractor
        self.with_labels = with_labels

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int):  # type: ignore[override]
        path, label = self.manifest[idx]
        array, sr = sf.read(path, dtype="float32")
        waveform = torch.from_numpy(array)
        if waveform.ndim > 1:  # stereo/multi-channel -> mono
            waveform = waveform.mean(dim=-1)
        x = self.feature_extractor(waveform, sr)
        if self.with_labels:
            return x, label
        return x


def build_local_audio_datasets(
    manifest_csv: str,
    path_column: str = "path",
    label_column: str = "label",
    eval_ratio: float = 0.2,
    seed: int = 42,
    feature_extractor: Optional[AudioFeatureExtractor] = None,
) -> Tuple[LocalAudioDataset, LocalAudioDataset, LocalAudioDataset]:
    """Loads a CSV manifest of local audio files and reuses it as both
    domains, same pattern as `build_audio_datasets` but with zero
    dependency on the HF Hub or `datasets` library.

    Args:
        manifest_csv: path to a CSV with (at least) a file-path column and
            an integer-label column, e.g.:
                path,label
                /data/clip001.wav,0
                /data/clip002.wav,2
        path_column / label_column: header names in that CSV.

    Returns (source_dataset, target_dataset, eval_dataset_raw).
    """
    feature_extractor = feature_extractor or AudioFeatureExtractor(
        backend="hubert", sampling_rate=_TARGET_SR
    )

    with open(manifest_csv, newline="") as f:
        reader = csv.DictReader(f)
        manifest: List[Tuple[str, int]] = [
            (row[path_column], int(row[label_column])) for row in reader
        ]

    rng = random.Random(seed)
    rng.shuffle(manifest)
    n_eval = int(len(manifest) * eval_ratio)
    eval_manifest, train_manifest = manifest[:n_eval], manifest[n_eval:]

    source_dataset = LocalAudioDataset(
        train_manifest, feature_extractor, with_labels=True
    )
    target_dataset = LocalAudioDataset(
        train_manifest, feature_extractor, with_labels=False
    )  # same audio, unlabeled
    eval_dataset_raw = LocalAudioDataset(
        eval_manifest, feature_extractor, with_labels=True
    )
    return source_dataset, target_dataset, eval_dataset_raw


def build_emodb_datasets(
    eval_ratio: float = 0.2,
    seed: int = 42,
    feature_extractor: Optional[AudioFeatureExtractor] = None,
):
    """Thin EmoDB-specific wrapper around `build_audio_datasets` (kept so
    existing calls don't need to change)."""
    return build_audio_datasets(
        dataset_name="renumics/emodb",
        split="train",
        audio_column="audio",
        label_column="emotion",
        eval_ratio=eval_ratio,
        seed=seed,
        feature_extractor=feature_extractor,
    )
