from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from datasets import load_dataset

from common import (
    build_few_shot_indices,
    canonicalize_class_name,
    ensure_dir,
    finish_wandb_run,
    init_wandb_run,
    load_config,
    log_wandb_artifact,
    save_json,
    save_rows_csv,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter Food101 to the configured 10 classes, export train/test images, "
            "and build few-shot train/validation manifests for each shot setting."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("assignments/assignment1/multimodal/configs/food101_clip.yaml"),
    )
    return parser.parse_args()


def filter_split_indices(labels: list[int], selected_original_ids: list[int]) -> list[int]:
    selected_set = set(selected_original_ids)
    return [index for index, label in enumerate(labels) if int(label) in selected_set]


def export_examples(
    dataset_split: Any,
    source_split: str,
    split_name: str,
    output_root: Path,
    selected_original_ids: list[int],
    original_to_new: dict[int, int],
    selected_classes: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    counters = {class_name: 0 for class_name in selected_classes}
    split_dir = ensure_dir(output_root / "images" / split_name)

    for source_index, example in enumerate(dataset_split):
        original_label_id = int(example["label"])
        if original_label_id not in original_to_new:
            continue

        label_id = original_to_new[original_label_id]
        label_name = selected_classes[label_id]
        class_dir = ensure_dir(split_dir / label_name)
        image_index = counters[label_name]
        filename = f"{split_name}_{label_name}_{image_index:05d}.jpg"
        image_path = class_dir / filename

        example["image"].convert("RGB").save(image_path, format="JPEG")
        counters[label_name] += 1

        rows.append(
            {
                "record_id": f"{source_split}:{source_index}",
                "source_split": source_split,
                "source_index": source_index,
                "split_name": split_name,
                "selected_label_id": label_id,
                "selected_label_name": label_name,
                "original_label_id": original_label_id,
                "image_path": str(image_path.relative_to(output_root)),
            }
        )

    return rows


def subset_rows(rows: list[dict[str, Any]], relative_indices: list[int]) -> list[dict[str, Any]]:
    return [rows[index] for index in relative_indices]


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config["splits"]["seed"]))
    wandb_run = init_wandb_run(
        config=config,
        job_type="preprocess",
        run_name=f"preprocess-{config['model']['id'].split('/')[-1]}",
        extra_config={"config_path": str(args.config)},
    )

    processed_root = ensure_dir(config["paths"]["processed_root"])
    dataset = load_dataset(config["dataset"]["name"])
    test_split_name = str(config["splits"]["test_split"])
    if "train" not in dataset or test_split_name not in dataset:
        raise ValueError(
            f"Expected dataset splits 'train' and '{test_split_name}', found {sorted(dataset.keys())}."
        )

    selected_classes = [
        canonicalize_class_name(name) for name in config["dataset"]["selected_classes"]
    ]
    available_class_names = dataset["train"].features["label"].names
    class_to_original = {
        canonicalize_class_name(name): index for index, name in enumerate(available_class_names)
    }
    missing_classes = [name for name in selected_classes if name not in class_to_original]
    if missing_classes:
        raise ValueError(f"Unknown selected classes: {missing_classes}")

    selected_original_ids = [class_to_original[name] for name in selected_classes]
    original_to_new = {
        original_label_id: new_label_id
        for new_label_id, original_label_id in enumerate(selected_original_ids)
    }

    train_rows = export_examples(
        dataset_split=dataset["train"],
        source_split="train",
        split_name="train_full",
        output_root=processed_root,
        selected_original_ids=selected_original_ids,
        original_to_new=original_to_new,
        selected_classes=selected_classes,
    )
    test_rows = export_examples(
        dataset_split=dataset[test_split_name],
        source_split=test_split_name,
        split_name="test",
        output_root=processed_root,
        selected_original_ids=selected_original_ids,
        original_to_new=original_to_new,
        selected_classes=selected_classes,
    )

    manifests_root = ensure_dir(processed_root / "manifests")
    fieldnames = [
        "record_id",
        "source_split",
        "source_index",
        "split_name",
        "selected_label_id",
        "selected_label_name",
        "original_label_id",
        "image_path",
    ]
    save_rows_csv(manifests_root / "train_full.csv", train_rows, fieldnames)
    save_rows_csv(manifests_root / "test.csv", test_rows, fieldnames)

    train_labels = [row["original_label_id"] for row in train_rows]
    val_per_class = int(config["splits"]["val_per_class"])
    split_summary: dict[str, Any] = {
        "dataset_name": config["dataset"]["name"],
        "selected_classes": selected_classes,
        "train_full_size": len(train_rows),
        "test_size": len(test_rows),
        "few_shot": {},
    }

    for shots in config["few_shot"]["num_shots"]:
        train_indices, val_indices = build_few_shot_indices(
            labels=train_labels,
            selected_original_ids=selected_original_ids,
            shots_per_class=int(shots),
            val_per_class=val_per_class,
            seed=int(config["splits"]["seed"]),
        )
        shot_dir = ensure_dir(manifests_root / f"fewshot_{int(shots)}")
        shot_train_rows = subset_rows(train_rows, train_indices)
        shot_val_rows = subset_rows(train_rows, val_indices)
        save_rows_csv(shot_dir / "train.csv", shot_train_rows, fieldnames)
        save_rows_csv(shot_dir / "val.csv", shot_val_rows, fieldnames)
        split_summary["few_shot"][str(int(shots))] = {
            "train_size": len(shot_train_rows),
            "val_size": len(shot_val_rows),
        }

    summary = {
        "config_path": str(args.config),
        "processed_root": str(processed_root),
        **split_summary,
    }
    save_json(
        processed_root / "summary.json",
        summary,
    )
    if wandb_run is not None:
        wandb_run.log(
            {
                "preprocess/train_full_size": len(train_rows),
                "preprocess/test_size": len(test_rows),
            }
        )
        for shots, info in split_summary["few_shot"].items():
            wandb_run.log(
                {
                    f"preprocess/fewshot_{shots}_train_size": info["train_size"],
                    f"preprocess/fewshot_{shots}_val_size": info["val_size"],
                }
            )
        log_wandb_artifact(
            wandb_run,
            artifact_name="food101_processed_manifests",
            artifact_type="dataset-manifests",
            path=processed_root / "manifests",
        )
        log_wandb_artifact(
            wandb_run,
            artifact_name="food101_processed_summary",
            artifact_type="metadata",
            path=processed_root / "summary.json",
        )
        finish_wandb_run(wandb_run, summary)
    print(f"Prepared data at {processed_root}")


if __name__ == "__main__":
    main()
