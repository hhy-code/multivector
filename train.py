import argparse
import os
import shutil
from pathlib import Path
from typing import Any

from transformers import TrainingArguments, set_seed, EarlyStoppingCallback

from dataset import RetrievalEvalDataset, TriplesCollator, TriplesDataset
from models import create_model_and_tokenizer
from trainer import MultiVectorTrainer
from utils.utils import load_config
from utils.dist import setup_ddp, cleanup_ddp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a multi-vector retriever with DDP.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the YAML training configuration.",
    )
    return parser.parse_args()


def create_datasets(
    config: dict[str, Any],
) -> tuple[TriplesDataset, RetrievalEvalDataset | None]:
    train_config = config["train"]
    train_dataset = TriplesDataset(
        triples=train_config["triples_path"],
        queries=train_config["queries_path"],
        corpus=train_config["corpus_path"],
        nway=train_config["nway"],
    )

    eval_config = config.get("eval")
    if eval_config is None:
        return train_dataset, None

    eval_dataset = RetrievalEvalDataset(
        qrels=eval_config["qrels_path"],
        queries=(
            train_dataset.queries
            if eval_config["queries_path"] == train_config["queries_path"]
            else eval_config["queries_path"]
        ),
        corpus=(
            train_dataset.corpus
            if eval_config["corpus_path"] == train_config["corpus_path"]
            else eval_config["corpus_path"]
        ),
    )
    return train_dataset, eval_dataset


def main() -> None:
    args = parse_args()
    config = load_config(str(args.config))
    training = config["training"]
    has_eval = config["data"].get("eval") is not None
    eval_steps = training.get("eval_steps", training.get("eval_and_save_steps", 500))
    save_steps = training.get("save_steps", training.get("eval_and_save_steps", 500))
    if training.get("fp16", False) and training.get("bf16", False):
        raise ValueError("fp16 and bf16 cannot both be enabled.")
    if save_steps < 1 or (has_eval and eval_steps < 1):
        raise ValueError("Evaluation/save intervals must be positive integer steps.")
    if has_eval and save_steps % eval_steps:
        raise ValueError("save_steps must be a multiple of eval_steps to restore the best model.")
    if training.get("early_stopping_patience", 8) < 1:
        raise ValueError("early_stopping_patience must be at least 1.")
    if training.get("gradient_accumulation_steps", 1) < 1:
        raise ValueError("gradient_accumulation_steps must be at least 1.")
    if training.get("max_grad_norm", 1.0) < 0:
        raise ValueError("max_grad_norm must be non-negative (0 disables clipping).")

    setup_ddp()
    seed = training.get("seed", 42)
    set_seed(seed)

    model_config = config["model"]
    model, tokenizer = create_model_and_tokenizer(
        model_config["model_name"],
        model_config,
    )

    data_config = config["data"]
    data_seed = data_config.get("seed", seed)
    train_dataset, eval_dataset = create_datasets(data_config)

    collator = TriplesCollator(tokenizer)

    os.environ["WANDB_PROJECT"] = training.get("wandb_project", "multi-vector")

    has_eval = eval_dataset is not None

    training_args = TrainingArguments(
        output_dir=training["output_dir"],
        num_train_epochs=training.get("num_train_epochs", 5),
        max_steps=training.get("max_steps", -1),
        per_device_train_batch_size=training["per_device_train_batch_size"],
        per_device_eval_batch_size=training.get(
            "per_device_eval_batch_size",
            training["per_device_train_batch_size"],
        ),
        learning_rate=training["learning_rate"],
        weight_decay=training.get("weight_decay", 0.01),
        lr_scheduler_type=training.get("lr_scheduler_type", "linear"),
        warmup_steps=training.get("warmup_steps", 5000),
        logging_steps=training.get("logging_steps", 50),
        eval_strategy="steps" if has_eval else "no",
        eval_steps=eval_steps,
        save_steps=save_steps,
        save_total_limit=training.get("save_total_limit", None),
        metric_for_best_model="ndcg@10",
        greater_is_better=True,
        bf16=training.get("bf16", False),
        fp16=training.get("fp16", False),
        gradient_accumulation_steps=training.get("gradient_accumulation_steps", 1),
        max_grad_norm=training.get("max_grad_norm", 1.0),
        report_to=training.get("report_to", "wandb"),
        run_name=training.get("run_name", model_config["model_name"]),
        logging_dir=training.get("logging_dir", "./logs"),
        seed=seed,
        data_seed=data_seed,
        dataloader_num_workers=training.get("dataloader_num_workers", 2),
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        dataloader_pin_memory=True,
        dataloader_drop_last=True,
        load_best_model_at_end=has_eval,
    )

    trainer = MultiVectorTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        gather_across_devices=training.get("gather_across_devices", True),
        score_margins=training.get("score_margins", False),
        callbacks=([
            EarlyStoppingCallback(
                early_stopping_patience=training.get("early_stopping_patience", 8),
                early_stopping_threshold=0.0
            )
        ] if has_eval else None),
    )

    if trainer.is_world_process_zero():
        output_dir = Path(training["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        config_path = output_dir / "training_config.yaml"
        if args.config.resolve() != config_path.resolve():
            shutil.copyfile(args.config, config_path)

    train_result = trainer.train(
        resume_from_checkpoint=training.get("resume_from_checkpoint")
    )
    if eval_dataset is not None:
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

    trainer.save_model()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    cleanup_ddp()


if __name__ == "__main__":
    main()
