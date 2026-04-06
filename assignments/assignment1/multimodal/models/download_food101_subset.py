from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from datasets import load_dataset


DEFAULT_CLASSES: Tuple[str, ...] = (
    "apple_pie",
    "bibimbap",
    "chicken_wings",
    "donuts",
    "eggs_benedict",
    "french_fries",
    "grilled_cheese_sandwich",
    "hamburger",
    "ice_cream",
    "pizza",
)


@dataclass(frozen=True)
class DownloadConfig:
    dataset_name: str
    output_root: Path
    class_names: Tuple[str, ...]
    image_format: str


def parse_args() -> DownloadConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Download the Food101 dataset, keep the selected 10 classes, and save "
            "the filtered subset under the assignment artifacts directory."
        )
    )
    parser.add_argument("--dataset-name", default="ethz/food101")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("assignments/assignment1/multimodal/artifacts/food101_subset"),
        help="Directory where the filtered Food101 subset will be written.",
    )
    parser.add_argument(
        "--class-name",
        action="append",
        dest="class_names",
        help="Optional class override. Repeat the flag to provide multiple Food101 classes.",
    )
    parser.add_argument(
        "--image-format",
        default="jpg",
        choices=("jpg", "png"),
        help="Image format to use when exporting the subset images.",
    )
    args = parser.parse_args()

    class_names = tuple(args.class_names) if args.class_names else DEFAULT_CLASSES
    return DownloadConfig(
        dataset_name=args.dataset_name,
        output_root=args.output_root,
        class_names=class_names,
        image_format=args.image_format,
    )


def canonicalize_class_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def humanize_class_name(name: str) -> str:
    return name.replace("_", " ")


def ensure_unique_class_names(class_names: Sequence[str]) -> Tuple[str, ...]:
    normalized = tuple(canonicalize_class_name(name) for name in class_names)
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"Class names must be unique, received: {list(class_names)}")
    return normalized


def resolve_class_ids(
    available_class_names: Sequence[str], selected_class_names: Sequence[str]
) -> Dict[int, str]:
    name_to_label = {
        canonicalize_class_name(class_name): index
        for index, class_name in enumerate(available_class_names)
    }
    missing = [name for name in selected_class_names if name not in name_to_label]
    if missing:
        raise ValueError(f"Unknown Food101 class names: {missing}")
    return {name_to_label[class_name]: class_name for class_name in selected_class_names}


def export_split(
    split_dataset: Any,
    split_name: str,
    selected_label_ids: Dict[int, str],
    output_root: Path,
    image_format: str,
) -> List[Dict[str, Any]]:
    split_root = output_root / split_name
    split_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    counters: Counter[str] = Counter()

    for example_index, example in enumerate(split_dataset):
        label_id = int(example["label"])
        if label_id not in selected_label_ids:
            continue

        class_name = selected_label_ids[label_id]
        class_root = split_root / class_name
        class_root.mkdir(parents=True, exist_ok=True)

        image_index = counters[class_name]
        filename = f"{split_name}_{class_name}_{image_index:05d}.{image_format}"
        image_path = class_root / filename

        image = example["image"].convert("RGB")
        save_format = "JPEG" if image_format == "jpg" else "PNG"
        image.save(image_path, format=save_format)
        counters[class_name] += 1

        rows.append(
            {
                "source_index": example_index,
                "split": split_name,
                "class_name": class_name,
                "class_display": humanize_class_name(class_name).title(),
                "label_id": label_id,
                "image_path": str(image_path.relative_to(output_root)),
            }
        )

    return rows


def write_metadata(
    output_root: Path,
    config: DownloadConfig,
    rows: Sequence[Dict[str, Any]],
) -> None:
    metadata_path = output_root / "metadata.json"
    summary_path = output_root / "summary.json"
    manifest_path = output_root / "manifest.jsonl"

    counts_by_split: Dict[str, Counter[str]] = {}
    for row in rows:
        split_counter = counts_by_split.setdefault(row["split"], Counter())
        split_counter[row["class_name"]] += 1

    summary = {
        "dataset_name": config.dataset_name,
        "output_root": str(output_root),
        "class_names": list(config.class_names),
        "class_names_readable": [humanize_class_name(name) for name in config.class_names],
        "image_format": config.image_format,
        "num_images": len(rows),
        "splits": {
            split_name: {
                "num_images": int(sum(counter.values())),
                "per_class": dict(sorted(counter.items())),
            }
            for split_name, counter in sorted(counts_by_split.items())
        },
    }

    metadata = {
        "config": {
            **asdict(config),
            "output_root": str(config.output_root),
            "class_names": list(config.class_names),
        },
        "summary": summary,
    }

    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def download_subset(config: DownloadConfig) -> Dict[str, Any]:
    config = DownloadConfig(
        dataset_name=config.dataset_name,
        output_root=config.output_root,
        class_names=ensure_unique_class_names(config.class_names),
        image_format=config.image_format,
    )

    dataset = load_dataset(config.dataset_name)
    if "train" not in dataset or "validation" not in dataset:
        available_splits = sorted(dataset.keys())
        raise ValueError(
            f"Expected Food101 splits 'train' and 'validation', found: {available_splits}"
        )

    available_class_names = dataset["train"].features["label"].names
    selected_label_ids = resolve_class_ids(available_class_names, config.class_names)
    config.output_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for split_name in ("train", "validation"):
        rows.extend(
            export_split(
                split_dataset=dataset[split_name],
                split_name=split_name,
                selected_label_ids=selected_label_ids,
                output_root=config.output_root,
                image_format=config.image_format,
            )
        )

    write_metadata(config.output_root, config, rows)
    return {
        "output_root": str(config.output_root),
        "num_images": len(rows),
        "class_names": list(config.class_names),
    }


def main() -> None:
    config = parse_args()
    result = download_subset(config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
