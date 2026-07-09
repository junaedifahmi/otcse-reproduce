"""
Entry point: load a YAML config, build TrainingArguments + OTCSEArguments
from it, and launch OTCSE training via the HF Trainer.

Usage:
    python train.py --config config.yaml
"""

import argparse
import os

import yaml
from transformers import TrainingArguments

from .dataset import PairedDomainDataset, build_emodb_datasets
from .evaluation import ClassificationEvalDataset, compute_metrics
from .sermodel import SERModel
from .trainer import (
    OTCSEArguments,
    OTCSETrainer,
)


def load_config(path: str):
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    training_args = TrainingArguments(**raw.get("training", {}))

    otcse_raw = raw.get("otcse", {})
    ot_raw = raw.get("optimal_transport", {})
    otcse_args = OTCSEArguments(
        mu=otcse_raw.get("mu", 0.7),
        nu=otcse_raw.get("nu", 0.7),
        ot_alpha=ot_raw.get("alpha", 0.1),
        ot_beta=ot_raw.get("beta", 0.001),
        ot_reg=ot_raw.get("reg", 0.5),
        ot_epsilon=ot_raw.get("epsilon", 0.01),
        ot_scale=ot_raw.get("scale", 5.0),
    )

    # wandb: TrainingArguments.report_to only names the backend; project/entity
    # are picked up from env vars by the wandb integration itself.
    wandb_raw = raw.get("wandb", {})
    if "wandb" in (training_args.report_to or []):
        if wandb_raw.get("project"):
            os.environ["WANDB_PROJECT"] = wandb_raw["project"]
        if wandb_raw.get("entity"):
            os.environ["WANDB_ENTITY"] = wandb_raw["entity"]

    return training_args, otcse_args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    training_args, otcse_args = load_config(args.config)

    # --- user-provided pieces -------------------------------------------------
    # Replace these with your actual model / datasets.
    model = SERModel()  # a modular model: model(x) -> (embedding, logits)
    # ---------------------------------------------------------------------------
    #
    # # EmoDB used as BOTH domains: labeled view = source, unlabeled view of the
    # same audio = target. Held-out slice is used for evaluation.
    source_dataset, target_dataset, eval_dataset_raw = build_emodb_datasets()
    # ---------------------------------------------------------------------------

    train_dataset = PairedDomainDataset(source_dataset, target_dataset)
    eval_dataset = ClassificationEvalDataset(eval_dataset_raw)

    trainer = OTCSETrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        otcse_args=otcse_args,
    )
    trainer.train()


if __name__ == "__main__":
    main()
