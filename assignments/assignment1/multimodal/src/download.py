from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the full ethz/food101 dataset, export all images to disk, "
            "and save labels as CSV and JSONL manifests."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/food101"),
        help="Directory where the exported dataset will be saved.",
    )
    parser.add_argument(
        "--dataset-name",
        default="ethz/food101",
        help="Hugging Face dataset identifier.",
    )
    parser.add_argument(
        "--image-format",
        choices=("jpg", "png"),
        default="jpg",
        help="Image format used when saving dataset images.",
    )
    return parser.parse_args()


def export_split(
    split_dataset: Any,
    split_name: str,
    output_dir: Path,
    class_names: list[str],
    image_format: str,
) -> list[dict[str, Any]]:
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    save_format = "JPEG" if image_format == "jpg" else "PNG"
    rows: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()

    for source_index, example in enumerate(split_dataset):
        label_id = int(example["label"])
        class_name = class_names[label_id]
        class_dir = split_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        image_index = counters[class_name]
        filename = f"{split_name}_{class_name}_{image_index:05d}.{image_format}"
        image_path = class_dir / filename

        example["image"].convert("RGB").save(image_path, format=save_format)
        counters[class_name] += 1

        rows.append(
            {
                "source_index": source_index,
                "split": split_name,
                "label_id": label_id,
                "label_name": class_name,
                "image_path": str(image_path.relative_to(output_dir)),
            }
        )

    return rows


def write_manifests(
    output_dir: Path,
    dataset_name: str,
    image_format: str,
    class_names: list[str],
    rows: list[dict[str, Any]],
) -> None:
    metadata = {
        "dataset_name": dataset_name,
        "num_classes": len(class_names),
        "class_names": class_names,
        "image_format": image_format,
        "num_images": len(rows),
        "splits": dict(Counter(row["split"] for row in rows)),
    }

    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    with (output_dir / "labels.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["source_index", "split", "label_id", "label_name", "image_path"],
        )
        writer.writeheader()
        writer.writerows(rows)

    with (output_dir / "labels.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.dataset_name)
    class_names = list(dataset["train"].features["label"].names)

    rows: list[dict[str, Any]] = []
    for split_name, split_dataset in dataset.items():
        rows.extend(
            export_split(
                split_dataset=split_dataset,
                split_name=split_name,
                output_dir=output_dir,
                class_names=class_names,
                image_format=args.image_format,
            )
        )

    write_manifests(
        output_dir=output_dir,
        dataset_name=args.dataset_name,
        image_format=args.image_format,
        class_names=class_names,
        rows=rows,
    )

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "num_images": len(rows),
                "num_classes": len(class_names),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
