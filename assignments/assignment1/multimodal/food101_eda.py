from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datasets import Dataset, concatenate_datasets


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
DEFAULT_SHOTS_PER_CLASS = 128
DEFAULT_DEV_PER_CLASS = 20
DEFAULT_SEED = 42


@dataclass(frozen=True)
class EDAConfig:
    cache_root: Path
    output_root: Path
    docs_output_root: Path
    class_names: Tuple[str, ...] = DEFAULT_CLASSES
    shots_per_class: int = DEFAULT_SHOTS_PER_CLASS
    dev_per_class: int = DEFAULT_DEV_PER_CLASS
    seed: int = DEFAULT_SEED


def parse_args() -> EDAConfig:
    parser = argparse.ArgumentParser(
        description="Generate Food101 EDA artifacts for the multimodal assignment page."
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path.home() / ".cache" / "huggingface" / "datasets" / "ethz___food101" / "default" / "0.0.0",
        help="Root directory containing the cached Food101 Arrow dataset shards.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("assignments/assignment1/multimodal/artifacts/dataset_eda"),
        help="Directory for generated data files and plots.",
    )
    parser.add_argument(
        "--docs-output-root",
        type=Path,
        default=Path("docs/assignment-1/multimodal/assets/dataset-eda"),
        help="Directory where the docs page expects static plot assets.",
    )
    parser.add_argument(
        "--class-name",
        action="append",
        dest="class_names",
        help="Optional class override. Repeat the flag to provide multiple Food101 classes.",
    )
    parser.add_argument("--shots-per-class", type=int, default=DEFAULT_SHOTS_PER_CLASS)
    parser.add_argument("--dev-per-class", type=int, default=DEFAULT_DEV_PER_CLASS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    class_names = tuple(args.class_names) if args.class_names else DEFAULT_CLASSES
    return EDAConfig(
        cache_root=args.cache_root,
        output_root=args.output_root,
        docs_output_root=args.docs_output_root,
        class_names=class_names,
        shots_per_class=args.shots_per_class,
        dev_per_class=args.dev_per_class,
        seed=args.seed,
    )


def humanize(name: str) -> str:
    return name.replace("_", " ")


def slugify(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def resolve_cache_snapshot(cache_root: Path) -> Path:
    if not cache_root.exists():
        raise FileNotFoundError(f"Food101 cache root does not exist: {cache_root}")

    snapshots = sorted([path for path in cache_root.iterdir() if path.is_dir()])
    if not snapshots:
        raise FileNotFoundError(f"No cached Food101 snapshot directories found in {cache_root}")
    return snapshots[-1]


def load_cached_food101(cache_snapshot: Path) -> Tuple[Dataset, Dataset, Dict[str, Any]]:
    dataset_info_path = cache_snapshot / "dataset_info.json"
    dataset_info = json.loads(dataset_info_path.read_text(encoding="utf-8"))

    train_parts = [Dataset.from_file(str(path)) for path in sorted(cache_snapshot.glob("food101-train-*.arrow"))]
    validation_parts = [
        Dataset.from_file(str(path)) for path in sorted(cache_snapshot.glob("food101-validation-*.arrow"))
    ]
    if not train_parts or not validation_parts:
        raise FileNotFoundError(f"Missing expected Arrow shards in {cache_snapshot}")

    return concatenate_datasets(train_parts), concatenate_datasets(validation_parts), dataset_info


def filter_by_labels(dataset: Dataset, selected_original_ids: Sequence[int]) -> Dataset:
    selected_set = set(selected_original_ids)
    keep_indices = [index for index, label in enumerate(dataset["label"]) if label in selected_set]
    return dataset.select(keep_indices)


def build_few_shot_indices(
    labels: Sequence[int],
    selected_original_ids: Sequence[int],
    shots_per_class: int,
    dev_per_class: int,
    seed: int,
) -> Tuple[List[int], List[int]]:
    grouped_indices: Dict[int, List[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        grouped_indices[int(label)].append(index)

    rng = random.Random(seed)
    train_indices: List[int] = []
    dev_indices: List[int] = []
    required_examples = shots_per_class + dev_per_class

    for original_label in selected_original_ids:
        label_indices = list(grouped_indices[original_label])
        rng.shuffle(label_indices)
        if len(label_indices) < required_examples:
            raise ValueError(
                f"Not enough samples for label {original_label}: need {required_examples}, found {len(label_indices)}"
            )
        train_indices.extend(label_indices[:shots_per_class])
        dev_indices.extend(label_indices[shots_per_class:required_examples])

    rng.shuffle(train_indices)
    rng.shuffle(dev_indices)
    return train_indices, dev_indices


def create_experiment_splits(
    train_dataset: Dataset,
    test_dataset: Dataset,
    selected_original_ids: Sequence[int],
    class_names: Sequence[str],
    shots_per_class: int,
    dev_per_class: int,
    seed: int,
) -> Dict[str, Dataset]:
    filtered_train = filter_by_labels(train_dataset, selected_original_ids)
    filtered_test = filter_by_labels(test_dataset, selected_original_ids)
    train_indices, dev_indices = build_few_shot_indices(
        labels=filtered_train["label"],
        selected_original_ids=selected_original_ids,
        shots_per_class=shots_per_class,
        dev_per_class=dev_per_class,
        seed=seed,
    )

    return {
        "few_shot_train": filtered_train.select(train_indices),
        "few_shot_dev": filtered_train.select(dev_indices),
        "test": filtered_test,
    }


def class_count_rows(
    datasets_by_split: Dict[str, Dataset],
    class_names: Sequence[str],
    original_to_selected: Dict[int, int],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for split_name, dataset in datasets_by_split.items():
        counts = Counter(original_to_selected[int(label)] for label in dataset["label"])
        for class_id, class_name in enumerate(class_names):
            rows.append(
                {
                    "split": split_name,
                    "split_display": split_name.replace("_", " ").title(),
                    "class_id": class_id,
                    "class_name": class_name,
                    "class_display": humanize(class_name).title(),
                    "count": int(counts.get(class_id, 0)),
                }
            )
    return rows


def image_stat_rows(
    datasets_by_split: Dict[str, Dataset],
    class_names: Sequence[str],
    original_to_selected: Dict[int, int],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for split_name, dataset in datasets_by_split.items():
        for sample_index, sample in enumerate(dataset):
            image = sample["image"]
            width, height = image.size
            mapped_label = original_to_selected[int(sample["label"])]
            class_name = class_names[mapped_label]
            rows.append(
                {
                    "split": split_name,
                    "split_display": split_name.replace("_", " ").title(),
                    "sample_index": sample_index,
                    "class_id": mapped_label,
                    "class_name": class_name,
                    "class_display": humanize(class_name).title(),
                    "width": int(width),
                    "height": int(height),
                    "aspect_ratio": float(width / height),
                    "area": int(width * height),
                }
            )
    return rows


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_json(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def copy_assets(plot_paths: Iterable[Path], docs_output_root: Path) -> None:
    docs_output_root.mkdir(parents=True, exist_ok=True)
    for path in plot_paths:
        shutil.copy2(path, docs_output_root / path.name)


def plot_class_distribution(dist_df: pd.DataFrame, path: Path) -> None:
    pivot = dist_df.pivot(index="class_display", columns="split_display", values="count")
    order = list(pivot.index)
    split_order = ["Few Shot Train", "Few Shot Dev", "Test"]
    colors = ["#233f5d", "#7d8ea3", "#4f8f74"]

    fig, ax = plt.subplots(figsize=(14, 6.5))
    x = np.arange(len(order))
    width = 0.25

    for offset, split_name, color in zip([-width, 0, width], split_order, colors):
        values = pivot[split_name].reindex(order).to_numpy()
        ax.bar(x + offset, values, width=width, label=split_name, color=color, alpha=0.9)

    ax.set_title("Food101 class distribution across experiment splits", fontsize=16, fontweight="bold")
    ax.set_xlabel("Food class")
    ax.set_ylabel("Samples")
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_split_composition(split_df: pd.DataFrame, path: Path) -> None:
    colors = ["#233f5d", "#8da2b6", "#4f8f74"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    axes[0].pie(
        split_df["count"],
        labels=split_df["split_display"],
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
        textprops={"fontsize": 10, "weight": "bold"},
    )
    axes[0].set_title("Split share of the active experiment subset", fontweight="bold")

    bars = axes[1].bar(split_df["split_display"], split_df["count"], color=colors, alpha=0.9)
    axes[1].set_title("Absolute sample count by split", fontweight="bold")
    axes[1].set_ylabel("Samples")
    axes[1].grid(axis="y", alpha=0.22)
    for bar, value, pct in zip(bars, split_df["count"], split_df["percentage"]):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{int(value)}\n{pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_balance_heatmap(dist_df: pd.DataFrame, balance_df: pd.DataFrame, path: Path) -> None:
    heatmap_df = dist_df.pivot(index="class_display", columns="split_display", values="count")
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={"width_ratios": [1.6, 1]})

    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=".0f",
        cmap="YlGnBu",
        linewidths=0.5,
        cbar_kws={"label": "Samples"},
        ax=axes[0],
    )
    axes[0].set_title("Class-by-split count heatmap", fontweight="bold")
    axes[0].set_xlabel("Split")
    axes[0].set_ylabel("Class")

    axes[1].axis("off")
    table = axes[1].table(
        cellText=balance_df.round(3).values.tolist(),
        colLabels=list(balance_df.columns),
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.08, 1.75)
    for column_index in range(len(balance_df.columns)):
        table[(0, column_index)].set_facecolor("#233f5d")
        table[(0, column_index)].set_text_props(color="white", weight="bold")
    axes[1].set_title("Balance summary", fontweight="bold", y=0.88)

    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_image_dimensions(image_df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    sns.boxplot(data=image_df, x="class_display", y="width", ax=axes[0, 0], color="#8da2b6")
    axes[0, 0].set_title("Width distribution by class", fontweight="bold")
    axes[0, 0].tick_params(axis="x", rotation=35)
    axes[0, 0].grid(axis="y", alpha=0.2)

    sns.boxplot(data=image_df, x="class_display", y="height", ax=axes[0, 1], color="#4f8f74")
    axes[0, 1].set_title("Height distribution by class", fontweight="bold")
    axes[0, 1].tick_params(axis="x", rotation=35)
    axes[0, 1].grid(axis="y", alpha=0.2)

    sns.violinplot(data=image_df, x="class_display", y="aspect_ratio", ax=axes[1, 0], color="#d8d3c4", inner="quart")
    axes[1, 0].set_title("Aspect ratio distribution", fontweight="bold")
    axes[1, 0].tick_params(axis="x", rotation=35)
    axes[1, 0].grid(axis="y", alpha=0.2)

    sns.histplot(data=image_df, x="area", hue="split_display", bins=36, ax=axes[1, 1], palette=["#233f5d", "#8da2b6", "#4f8f74"])
    axes[1, 1].set_title("Image area distribution by split", fontweight="bold")
    axes[1, 1].grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_dimension_scatter(image_df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(
        image_df["width"],
        image_df["height"],
        c=image_df["aspect_ratio"],
        cmap="viridis",
        s=26,
        alpha=0.55,
        linewidths=0,
    )
    ax.plot([0, 1200], [0, 1200], linestyle="--", color="#233f5d", alpha=0.4, label="square ratio")
    ax.set_title("Width vs height across selected Food101 images", fontsize=15, fontweight="bold")
    ax.set_xlabel("Width (px)")
    ax.set_ylabel("Height (px)")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Aspect ratio")

    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_single_sample_grid(
    test_dataset: Dataset,
    class_names: Sequence[str],
    original_to_selected: Dict[int, int],
    path: Path,
) -> None:
    selected_examples: Dict[int, Any] = {}
    for sample in test_dataset:
        mapped_label = original_to_selected[int(sample["label"])]
        if mapped_label not in selected_examples:
            selected_examples[mapped_label] = sample["image"]
        if len(selected_examples) == len(class_names):
            break

    fig, axes = plt.subplots(2, 5, figsize=(16, 8.5))
    for class_id, ax in enumerate(axes.flatten()):
        ax.imshow(selected_examples[class_id])
        ax.set_title(humanize(class_names[class_id]).title(), fontsize=11, fontweight="bold")
        ax.axis("off")

    fig.suptitle("One test sample per selected Food101 class", fontsize=16, fontweight="bold", y=0.98)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_multi_sample_grid(
    test_dataset: Dataset,
    class_names: Sequence[str],
    original_to_selected: Dict[int, int],
    path: Path,
    samples_per_class: int = 3,
) -> None:
    class_examples: Dict[int, List[Any]] = {class_id: [] for class_id in range(len(class_names))}
    for sample in test_dataset:
        mapped_label = original_to_selected[int(sample["label"])]
        if len(class_examples[mapped_label]) < samples_per_class:
            class_examples[mapped_label].append(sample["image"])
        if all(len(images) >= samples_per_class for images in class_examples.values()):
            break

    fig, axes = plt.subplots(len(class_names), samples_per_class, figsize=(10.5, 20))
    for class_id, class_name in enumerate(class_names):
        for sample_index in range(samples_per_class):
            ax = axes[class_id, sample_index]
            ax.imshow(class_examples[class_id][sample_index])
            if sample_index == 0:
                ax.set_ylabel(humanize(class_name).title(), fontsize=10, fontweight="bold", labelpad=14)
            ax.axis("off")

    fig.suptitle("Test gallery: three examples per class", fontsize=16, fontweight="bold", y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_summary(
    config: EDAConfig,
    cache_snapshot: Path,
    dataset_info: Dict[str, Any],
    split_df: pd.DataFrame,
    dist_df: pd.DataFrame,
    image_df: pd.DataFrame,
) -> Dict[str, Any]:
    balance_rows: List[Dict[str, Any]] = []
    for split_name in ["few_shot_train", "few_shot_dev", "test"]:
        counts = dist_df.loc[dist_df["split"] == split_name, "count"].to_numpy()
        imbalance_ratio = float((counts.max() - counts.min()) / counts.max()) if counts.size else 0.0
        balance_rows.append(
            {
                "split": split_name,
                "split_display": split_name.replace("_", " ").title(),
                "min_count": int(counts.min()),
                "max_count": int(counts.max()),
                "mean_count": float(counts.mean()),
                "std_count": float(counts.std()),
                "imbalance_ratio": imbalance_ratio,
            }
        )

    per_class_totals = (
        dist_df.groupby(["class_name", "class_display"], as_index=False)["count"]
        .sum()
        .sort_values("count", ascending=False)
    )

    image_summary = {
        "width_min": int(image_df["width"].min()),
        "width_max": int(image_df["width"].max()),
        "height_min": int(image_df["height"].min()),
        "height_max": int(image_df["height"].max()),
        "width_mean": float(image_df["width"].mean()),
        "height_mean": float(image_df["height"].mean()),
        "aspect_ratio_mean": float(image_df["aspect_ratio"].mean()),
        "aspect_ratio_std": float(image_df["aspect_ratio"].std()),
        "aspect_ratio_min": float(image_df["aspect_ratio"].min()),
        "aspect_ratio_max": float(image_df["aspect_ratio"].max()),
        "area_mean": float(image_df["area"].mean()),
        "area_median": float(image_df["area"].median()),
    }

    total_subset_count = int(dataset_info["splits"]["train"]["num_examples"] * len(config.class_names) / 101)
    total_subset_count += int(dataset_info["splits"]["validation"]["num_examples"] * len(config.class_names) / 101)

    findings = [
        "All three active splits are perfectly balanced across the 10 selected classes.",
        "The test set dominates the active experiment subset, which is appropriate because evaluation uses the full held-out split.",
        "Image geometry is diverse: aspect ratios vary materially, so center-crop-only preprocessing would discard non-trivial content.",
        "Each selected class contributes exactly 1,000 images in the original filtered subset, eliminating label-frequency bias from class choice.",
    ]

    return {
        "config": {
            **asdict(config),
            "cache_snapshot": str(cache_snapshot),
            "class_names": list(config.class_names),
        },
        "dataset": {
            "name": "ethz/food101",
            "builder_name": dataset_info.get("builder_name"),
            "selected_classes": list(config.class_names),
            "selected_classes_display": [humanize(name).title() for name in config.class_names],
            "full_dataset_train_count": int(dataset_info["splits"]["train"]["num_examples"]),
            "full_dataset_test_count": int(dataset_info["splits"]["validation"]["num_examples"]),
            "selected_subset_total_count": total_subset_count,
            "selected_subset_per_class_count": 1000,
            "active_experiment_total_count": int(split_df["count"].sum()),
        },
        "split_summary": split_df.to_dict(orient="records"),
        "balance_summary": balance_rows,
        "per_class_totals": per_class_totals.to_dict(orient="records"),
        "image_summary": image_summary,
        "findings": findings,
    }


def generate_eda(config: EDAConfig) -> Dict[str, Any]:
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.titleweight"] = "bold"

    cache_snapshot = resolve_cache_snapshot(config.cache_root)
    train_dataset, test_dataset, dataset_info = load_cached_food101(cache_snapshot)

    all_class_names = dataset_info["features"]["label"]["names"]
    name_to_original_id = {slugify(name): index for index, name in enumerate(all_class_names)}
    selected_class_names = [slugify(name) for name in config.class_names]
    missing = [name for name in selected_class_names if name not in name_to_original_id]
    if missing:
        raise ValueError(f"Unknown Food101 classes requested: {missing}")

    selected_original_ids = [name_to_original_id[name] for name in selected_class_names]
    original_to_selected = {original_id: new_id for new_id, original_id in enumerate(selected_original_ids)}

    datasets_by_split = create_experiment_splits(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        selected_original_ids=selected_original_ids,
        class_names=selected_class_names,
        shots_per_class=config.shots_per_class,
        dev_per_class=config.dev_per_class,
        seed=config.seed,
    )

    config.output_root.mkdir(parents=True, exist_ok=True)
    config.docs_output_root.mkdir(parents=True, exist_ok=True)

    dist_df = pd.DataFrame(class_count_rows(datasets_by_split, selected_class_names, original_to_selected))
    image_df = pd.DataFrame(image_stat_rows(datasets_by_split, selected_class_names, original_to_selected))

    split_df = (
        dist_df.groupby(["split", "split_display"], as_index=False)["count"]
        .sum()
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )
    split_df["percentage"] = split_df["count"] / split_df["count"].sum() * 100

    balance_df = pd.DataFrame(
        [
            {
                "Split": split_name.replace("_", " ").title(),
                "Min": int(values.min()),
                "Max": int(values.max()),
                "Mean": float(values.mean()),
                "Std": float(values.std()),
                "Imbalance Ratio": float((values.max() - values.min()) / values.max()) if values.size else 0.0,
            }
            for split_name in ["few_shot_train", "few_shot_dev", "test"]
            for values in [dist_df.loc[dist_df["split"] == split_name, "count"].to_numpy()]
        ]
    )

    plot_paths = [
        config.output_root / "class_distribution.png",
        config.output_root / "split_composition.png",
        config.output_root / "balance_heatmap.png",
        config.output_root / "image_dimensions.png",
        config.output_root / "dimension_scatter.png",
        config.output_root / "sample_grid.png",
        config.output_root / "class_gallery.png",
    ]

    plot_class_distribution(dist_df, plot_paths[0])
    plot_split_composition(split_df, plot_paths[1])
    plot_balance_heatmap(dist_df, balance_df, plot_paths[2])
    plot_image_dimensions(image_df, plot_paths[3])
    plot_dimension_scatter(image_df, plot_paths[4])
    plot_single_sample_grid(datasets_by_split["test"], selected_class_names, original_to_selected, plot_paths[5])
    plot_multi_sample_grid(datasets_by_split["test"], selected_class_names, original_to_selected, plot_paths[6])

    save_dataframe(dist_df, config.output_root / "class_distribution.csv")
    save_dataframe(split_df, config.output_root / "split_summary.csv")
    save_dataframe(balance_df, config.output_root / "balance_summary.csv")
    save_dataframe(image_df, config.output_root / "image_statistics.csv")

    summary = build_summary(
        config=config,
        cache_snapshot=cache_snapshot,
        dataset_info=dataset_info,
        split_df=split_df,
        dist_df=dist_df,
        image_df=image_df,
    )
    save_json(summary, config.output_root / "eda_summary.json")
    copy_assets(plot_paths, config.docs_output_root)
    shutil.copy2(config.output_root / "eda_summary.json", config.docs_output_root / "eda_summary.json")
    shutil.copy2(config.output_root / "class_distribution.csv", config.docs_output_root / "class_distribution.csv")
    shutil.copy2(config.output_root / "split_summary.csv", config.docs_output_root / "split_summary.csv")
    shutil.copy2(config.output_root / "balance_summary.csv", config.docs_output_root / "balance_summary.csv")

    return summary


def main() -> None:
    config = parse_args()
    summary = generate_eda(config)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
