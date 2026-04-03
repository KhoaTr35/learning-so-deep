import argparse
import copy
import csv
import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from transformers import AutoProcessor, CLIPModel

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_PROMPT_TEMPLATES = (
    "a photo of {}.",
    "a close-up photo of {}.",
    "a photo of a plate of {}.",
    "a food photo of {}.",
    "a restaurant photo of {}.",
)


@dataclass
class ExperimentConfig:
    model_id: str = "openai/clip-vit-base-patch32"
    dataset_name: str = "ethz/food101"
    top_k: int = 10
    class_names: Optional[List[str]] = None
    shots_per_class: int = 8
    val_per_class: int = 20
    batch_size: int = 16
    num_workers: int = 0
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42
    device: Optional[str] = None
    output_root: Path = Path("outputs/food101_experiments")
    prompt_templates: Tuple[str, ...] = field(
        default_factory=lambda: tuple(DEFAULT_PROMPT_TEMPLATES)
    )

    def to_serializable_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["output_root"] = str(self.output_root)
        payload["prompt_templates"] = list(self.prompt_templates)
        return payload


class Food101SubsetDataset(Dataset):
    def __init__(
        self,
        dataset_split: Any,
        original_to_new: Dict[int, int],
        class_names: Sequence[str],
        split_name: str,
    ) -> None:
        self.dataset_split = dataset_split
        self.original_to_new = original_to_new
        self.class_names = list(class_names)
        self.split_name = split_name

    def __len__(self) -> int:
        return len(self.dataset_split)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        example = self.dataset_split[index]
        image = example["image"]
        if not isinstance(image, Image.Image):
            raise TypeError(f"Expected a PIL image, got {type(image)!r}")

        label_id = self.original_to_new[example["label"]]
        return {
            "index": index,
            "image": image.convert("RGB"),
            "label": label_id,
            "label_name": self.class_names[label_id],
            "split_name": self.split_name,
        }


###################
#      UTILS      #
###################
def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Run a CLIP-based Food101 experiment that compares zero-shot and "
            "few-shot classification on the top 10 classes."
        )
    )
    parser.add_argument("--model-id", default="openai/clip-vit-base-patch32")
    parser.add_argument("--dataset-name", default="food101")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--class-name",
        action="append",
        dest="class_names",
        help=(
            "Optional explicit Food101 class name. Repeat the flag to define a "
            "custom subset, for example --class-name apple_pie."
        ),
    )
    parser.add_argument("--shots-per-class", type=int, default=8)
    parser.add_argument("--val-per-class", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/food101_experiments"),
    )
    parser.add_argument(
        "--prompt-template",
        action="append",
        dest="prompt_templates",
        help=(
            "Optional prompt template for zero-shot CLIP prompts. Use {} as the "
            "class placeholder. Repeat the flag for multiple templates."
        ),
    )
    args = parser.parse_args()

    prompt_templates = (
        tuple(args.prompt_templates)
        if args.prompt_templates
        else tuple(DEFAULT_PROMPT_TEMPLATES)
    )
    class_names = list(args.class_names) if args.class_names else None

    return ExperimentConfig(
        model_id=args.model_id,
        dataset_name=args.dataset_name,
        top_k=args.top_k,
        class_names=class_names,
        shots_per_class=args.shots_per_class,
        val_per_class=args.val_per_class,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
        output_root=args.output_root,
        prompt_templates=prompt_templates,
    )


def canonicalize_class_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def humanize_class_name(name: str) -> str:
    return name.replace("_", " ")


def resolve_device(requested_device: Optional[str] = None) -> torch.device:
    if requested_device:
        return torch.device(requested_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_config(config: ExperimentConfig) -> None:
    if config.top_k <= 0:
        raise ValueError("--top-k must be positive.")
    if config.shots_per_class <= 0:
        raise ValueError("--shots-per-class must be positive.")
    if config.val_per_class < 0:
        raise ValueError("--val-per-class must be zero or positive.")
    if config.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if config.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if config.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive.")
    if config.weight_decay < 0:
        raise ValueError("--weight-decay must be zero or positive.")
    if not config.prompt_templates:
        raise ValueError("At least one zero-shot prompt template is required.")
    if any("{}" not in template for template in config.prompt_templates):
        raise ValueError("Every --prompt-template must include a {} placeholder.")


def create_output_dir(output_root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_root / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def save_rows_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_confusion_matrix_csv(
    path: Path, matrix: np.ndarray, class_names: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    readable_names = [humanize_class_name(name) for name in class_names]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_label"] + readable_names)
        for class_name, row in zip(readable_names, matrix.tolist()):
            writer.writerow([class_name] + row)


def save_confusion_matrix_plot(
    path: Path, matrix: np.ndarray, class_names: Sequence[str], title: str
) -> None:
    readable_names = [humanize_class_name(name) for name in class_names]
    figure_width = max(8, len(class_names) * 1.1)
    figure_height = max(6, len(class_names) * 0.9)
    fig, ax = plt.subplots(figsize=(figure_width, figure_height))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(len(readable_names)))
    ax.set_yticks(np.arange(len(readable_names)))
    ax.set_xticklabels(readable_names, rotation=45, ha="right")
    ax.set_yticklabels(readable_names)

    threshold = matrix.max() / 2 if matrix.size else 0.0
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            color = "white" if matrix[row_index, column_index] > threshold else "black"
            ax.text(
                column_index,
                row_index,
                int(matrix[row_index, column_index]),
                ha="center",
                va="center",
                color=color,
                fontsize=8,
            )

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def image_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "indices": torch.tensor([item["index"] for item in batch], dtype=torch.long),
        "images": [item["image"] for item in batch],
        "labels": torch.tensor([item["label"] for item in batch], dtype=torch.long),
        "label_names": [item["label_name"] for item in batch],
        "split_names": [item["split_name"] for item in batch],
    }

#################################
#        Data processing        #
#################################

def select_classes(
    available_class_names: Sequence[str], config: ExperimentConfig
) -> List[str]:
    available_set = {canonicalize_class_name(name) for name in available_class_names}
    if config.class_names:
        selected = [canonicalize_class_name(name) for name in config.class_names]
    else:
        selected = sorted(available_set)[: config.top_k]

    if len(selected) != len(set(selected)):
        raise ValueError("Class names must be unique.")
    if len(selected) == 0:
        raise ValueError("At least one class must be selected.")
    if len(selected) > len(available_set):
        raise ValueError("Requested more classes than the dataset provides.")

    missing = sorted(set(selected) - available_set)
    if missing:
        raise ValueError(f"Unknown Food101 classes requested: {missing}")
    return selected


def filter_dataset_split(dataset_split: Any, selected_original_ids: Sequence[int]) -> Any:
    selected_set = set(selected_original_ids)
    labels = dataset_split["label"]
    keep_indices = [index for index, label in enumerate(labels) if label in selected_set]
    return dataset_split.select(keep_indices)


def build_few_shot_indices(
    labels: Sequence[int],
    selected_original_ids: Sequence[int],
    shots_per_class: int,
    val_per_class: int,
    seed: int,
) -> Tuple[List[int], List[int]]:
    grouped_indices: Dict[int, List[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        grouped_indices[label].append(index)

    rng = random.Random(seed)
    train_indices: List[int] = []
    val_indices: List[int] = []
    required_examples = shots_per_class + val_per_class

    for original_label in selected_original_ids:
        label_indices = list(grouped_indices[original_label])
        rng.shuffle(label_indices)
        if len(label_indices) < required_examples:
            raise ValueError(
                "Not enough samples to build the few-shot split for label "
                f"{original_label}: required {required_examples}, found {len(label_indices)}."
            )
        train_indices.extend(label_indices[:shots_per_class])
        val_indices.extend(label_indices[shots_per_class:required_examples])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def load_food101_datasets(
    config: ExperimentConfig,
) -> Tuple[Dict[str, Dataset], List[str], Dict[str, Any]]:
    dataset = load_dataset(config.dataset_name)
    available_splits = sorted(dataset.keys())
    if "train" not in dataset or "validation" not in dataset:
        raise ValueError(
            "This pipeline expects Food101 to expose 'train' and 'validation' "
            f"splits, found: {available_splits}"
        )

    available_class_names = dataset["train"].features["label"].names
    selected_class_names = select_classes(available_class_names, config)

    name_to_original_id = {
        canonicalize_class_name(name): index
        for index, name in enumerate(available_class_names)
    }
    selected_original_ids = [name_to_original_id[name] for name in selected_class_names]
    original_to_new = {
        original_id: new_id for new_id, original_id in enumerate(selected_original_ids)
    }

    filtered_train = filter_dataset_split(dataset["train"], selected_original_ids)
    filtered_test = filter_dataset_split(
        dataset["validation"], selected_original_ids
    )
    few_shot_train_indices, few_shot_dev_indices = build_few_shot_indices(
        labels=filtered_train["label"],
        selected_original_ids=selected_original_ids,
        shots_per_class=config.shots_per_class,
        val_per_class=config.val_per_class,
        seed=config.seed,
    )

    train_dataset = Food101SubsetDataset(
        dataset_split=filtered_train.select(few_shot_train_indices),
        original_to_new=original_to_new,
        class_names=selected_class_names,
        split_name="few_shot_train",
    )
    dev_dataset = Food101SubsetDataset(
        dataset_split=filtered_train.select(few_shot_dev_indices),
        original_to_new=original_to_new,
        class_names=selected_class_names,
        split_name="few_shot_dev",
    )
    evaluation_dataset = Food101SubsetDataset(
        dataset_split=filtered_test,
        original_to_new=original_to_new,
        class_names=selected_class_names,
        split_name="test",
    )

    dataset_summary = {
        "dataset_name": config.dataset_name,
        "available_splits": available_splits,
        "selected_class_names": selected_class_names,
        "selected_class_names_readable": [
            humanize_class_name(name) for name in selected_class_names
        ],
        "shots_per_class": config.shots_per_class,
        "dev_per_class": config.val_per_class,
        "few_shot_train_size": len(train_dataset),
        "few_shot_dev_size": len(dev_dataset),
        "held_out_test_size": len(evaluation_dataset),
        "few_shot_train_size_per_class": config.shots_per_class,
        "few_shot_dev_size_per_class": config.val_per_class,
        "evaluation_split_name": "test",
    }

    return (
        {
            "few_shot_train": train_dataset,
            "few_shot_dev": dev_dataset,
            "test": evaluation_dataset,
        },
        selected_class_names,
        dataset_summary,
    )


####################
#    Dataloader    #
####################
def create_image_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=image_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )


def extract_image_features(
    model: CLIPModel,
    processor: AutoProcessor,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    feature_batches: List[torch.Tensor] = []
    label_batches: List[torch.Tensor] = []
    index_batches: List[torch.Tensor] = []

    model.eval()
    with torch.inference_mode():
        for batch in dataloader:
            inputs = processor(images=batch["images"], return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
            outputs = model.get_image_features(pixel_values=pixel_values)
            if hasattr(outputs, "pooler_output"):
                image_features = outputs.pooler_output
            elif isinstance(outputs, (list, tuple)):
                image_features = outputs[0]
            else:
                image_features = outputs
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            feature_batches.append(image_features.cpu())
            label_batches.append(batch["labels"].cpu())
            index_batches.append(batch["indices"].cpu())

    return (
        torch.cat(feature_batches, dim=0),
        torch.cat(label_batches, dim=0),
        torch.cat(index_batches, dim=0),
    )


def build_zero_shot_text_features(
    model: CLIPModel,
    processor: AutoProcessor,
    class_names: Sequence[str],
    prompt_templates: Sequence[str],
    device: torch.device,
) -> torch.Tensor:
    text_feature_batches: List[torch.Tensor] = []
    model.eval()

    with torch.inference_mode():
        for class_name in class_names:
            prompts = [
                template.format(humanize_class_name(class_name))
                for template in prompt_templates
            ]
            text_inputs = processor(
                text=prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            text_inputs = {
                key: value.to(device) for key, value in text_inputs.items()
            }
            # Lấy text features
            text_outputs = model.get_text_features(**text_inputs)

            # Kiểm tra và trích xuất Tensor từ Object
            if hasattr(text_outputs, "pooler_output"):
                text_features = text_outputs.pooler_output
            elif isinstance(text_outputs, (list, tuple)):
                text_features = text_outputs[0]
            else:
                text_features = text_outputs
                
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            averaged_feature = text_features.mean(dim=0)
            averaged_feature = averaged_feature / averaged_feature.norm()
            text_feature_batches.append(averaged_feature.cpu())

    return torch.stack(text_feature_batches, dim=0)


def compute_metrics(
    true_labels: Sequence[int],
    predicted_labels: Sequence[int],
    class_names: Sequence[str],
) -> Tuple[Dict[str, float], np.ndarray, Dict[str, Any]]:
    labels = list(range(len(class_names)))
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        average="macro",
        zero_division=0,
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        average="weighted",
        zero_division=0,
    )
    matrix = confusion_matrix(true_labels, predicted_labels, labels=labels)
    report = classification_report(
        true_labels,
        predicted_labels,
        labels=labels,
        target_names=[humanize_class_name(name) for name in class_names],
        zero_division=0,
        output_dict=True,
    )
    metrics = {
        "accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "balanced_accuracy": float(
            balanced_accuracy_score(true_labels, predicted_labels)
        ),
        "macro_precision": float(precision_macro),
        "macro_recall": float(recall_macro),
        "macro_f1": float(f1_macro),
        "weighted_precision": float(precision_weighted),
        "weighted_recall": float(recall_weighted),
        "weighted_f1": float(f1_weighted),
    }
    return metrics, matrix, report


def evaluate_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    indices: torch.Tensor,
    class_names: Sequence[str],
    method_name: str,
) -> Dict[str, Any]:
    probabilities = torch.softmax(logits, dim=-1)
    confidence_values, predictions = probabilities.max(dim=-1)
    true_labels = labels.cpu().tolist()
    predicted_labels = predictions.cpu().tolist()
    sample_indices = indices.cpu().tolist()
    metrics, matrix, report = compute_metrics(
        true_labels=true_labels,
        predicted_labels=predicted_labels,
        class_names=class_names,
    )

    prediction_rows = []
    for sample_index, true_label, predicted_label, confidence in zip(
        sample_indices,
        true_labels,
        predicted_labels,
        confidence_values.cpu().tolist(),
    ):
        prediction_rows.append(
            {
                "sample_index": int(sample_index),
                "method": method_name,
                "true_label_id": int(true_label),
                "true_label_name": class_names[true_label],
                "predicted_label_id": int(predicted_label),
                "predicted_label_name": class_names[predicted_label],
                "confidence": float(confidence),
            }
        )

    return {
        "metrics": metrics,
        "confusion_matrix": matrix,
        "classification_report": report,
        "predictions": prediction_rows,
    }


def save_evaluation_artifacts(
    output_dir: Path,
    method_name: str,
    evaluation: Dict[str, Any],
    class_names: Sequence[str],
) -> None:
    method_dir = output_dir / method_name
    save_json(method_dir / "metrics.json", evaluation["metrics"])
    save_json(
        method_dir / "classification_report.json",
        evaluation["classification_report"],
    )
    save_rows_csv(
        method_dir / "predictions.csv",
        evaluation["predictions"],
        fieldnames=[
            "sample_index",
            "method",
            "true_label_id",
            "true_label_name",
            "predicted_label_id",
            "predicted_label_name",
            "confidence",
        ],
    )
    save_confusion_matrix_csv(
        method_dir / "confusion_matrix.csv",
        evaluation["confusion_matrix"],
        class_names,
    )
    save_confusion_matrix_plot(
        method_dir / "confusion_matrix.png",
        evaluation["confusion_matrix"],
        class_names,
        title=f"{method_name.replace('_', ' ').title()} Confusion Matrix",
    )


def train_linear_probe(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    dev_features: torch.Tensor,
    dev_labels: torch.Tensor,
    config: ExperimentConfig,
    class_names: Sequence[str],
    device: torch.device,
    output_dir: Path,
) -> Tuple[nn.Linear, List[Dict[str, Any]], Dict[str, Any]]:
    classifier = nn.Linear(train_features.shape[1], len(class_names)).to(device)
    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    train_loader = DataLoader(
        TensorDataset(train_features, train_labels),
        batch_size=min(config.batch_size * 4, len(train_features)),
        shuffle=True,
    )
    has_dev = len(dev_features) > 0
    history_rows: List[Dict[str, Any]] = []
    best_state = copy.deepcopy(classifier.state_dict())
    best_metrics: Dict[str, Any] = {}
    best_macro_f1 = float("-inf")

    for epoch in range(1, config.epochs + 1):
        classifier.train()
        running_loss = 0.0
        seen_examples = 0

        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = classifier(batch_features)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

            batch_size = batch_labels.size(0)
            running_loss += loss.item() * batch_size
            seen_examples += batch_size

        epoch_record: Dict[str, Any] = {
            "epoch": epoch,
            "train_loss": running_loss / max(seen_examples, 1),
        }

        if has_dev:
            dev_logits = predict_linear_probe_logits(
                classifier=classifier,
                features=dev_features,
                batch_size=config.batch_size * 8,
                device=device,
            )
            dev_loss = criterion(dev_logits, dev_labels).item()
            dev_eval = evaluate_logits(
                logits=dev_logits,
                labels=dev_labels,
                indices=torch.arange(len(dev_labels)),
                class_names=class_names,
                method_name="few_shot_dev",
            )
            epoch_record.update(
                {
                    "dev_loss": dev_loss,
                    "dev_accuracy": dev_eval["metrics"]["accuracy"],
                    "dev_macro_f1": dev_eval["metrics"]["macro_f1"],
                }
            )

            if dev_eval["metrics"]["macro_f1"] > best_macro_f1:
                best_macro_f1 = dev_eval["metrics"]["macro_f1"]
                best_metrics = dev_eval["metrics"]
                best_state = {
                    key: value.detach().cpu()
                    for key, value in classifier.state_dict().items()
                }
        else:
            best_state = {
                key: value.detach().cpu()
                for key, value in classifier.state_dict().items()
            }

        history_rows.append(epoch_record)

    classifier.load_state_dict(best_state)

    checkpoint = {
        "model_type": "linear_probe",
        "feature_dim": int(train_features.shape[1]),
        "num_classes": len(class_names),
        "class_names": list(class_names),
        "config": config.to_serializable_dict(),
        "state_dict": {
            key: value.detach().cpu() for key, value in classifier.state_dict().items()
        },
        "best_dev_metrics": best_metrics,
    }
    torch.save(checkpoint, output_dir / "few_shot_linear_probe.pt")

    return classifier, history_rows, best_metrics


def predict_linear_probe_logits(
    classifier: nn.Linear,
    features: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    logits_batches: List[torch.Tensor] = []
    classifier.eval()
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            batch_features = features[start : start + batch_size].to(device)
            logits = classifier(batch_features)
            logits_batches.append(logits.cpu())
    return torch.cat(logits_batches, dim=0)


def run_experiment(config: ExperimentConfig) -> Dict[str, Any]:
    validate_config(config)
    set_seed(config.seed)
    device = resolve_device(config.device)
    output_dir = create_output_dir(config.output_root)
    save_json(output_dir / "config.json", config.to_serializable_dict())

    print(f"Loading dataset {config.dataset_name!r}...")
    datasets_by_split, class_names, dataset_summary = load_food101_datasets(config)
    dataset_summary["device"] = str(device)
    save_json(output_dir / "dataset_summary.json", dataset_summary)

    print(f"Loading CLIP model {config.model_id!r} on {device}...")
    processor = AutoProcessor.from_pretrained(config.model_id)
    model = CLIPModel.from_pretrained(config.model_id)
    model.to(device)
    model.eval()

    print("Building dataloaders...")
    dataloaders = {
        split_name: create_image_dataloader(
            dataset=dataset_split,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            shuffle=False,
        )
        for split_name, dataset_split in datasets_by_split.items()
    }

    print("Extracting image features...")
    split_features: Dict[str, torch.Tensor] = {}
    split_labels: Dict[str, torch.Tensor] = {}
    split_indices: Dict[str, torch.Tensor] = {}
    for split_name, dataloader in dataloaders.items():
        features, labels, indices = extract_image_features(
            model=model,
            processor=processor,
            dataloader=dataloader,
            device=device,
        )
        split_features[split_name] = features
        split_labels[split_name] = labels
        split_indices[split_name] = indices

    print("Building zero-shot classifier...")
    text_features = build_zero_shot_text_features(
        model=model,
        processor=processor,
        class_names=class_names,
        prompt_templates=config.prompt_templates,
        device=device,
    )
    torch.save(
        {
            "class_names": class_names,
            "prompt_templates": list(config.prompt_templates),
            "text_features": text_features,
        },
        output_dir / "zero_shot_text_features.pt",
    )

    logit_scale = float(model.logit_scale.exp().detach().cpu().item())
    zero_shot_logits = logit_scale * (split_features["test"] @ text_features.T)
    zero_shot_eval = evaluate_logits(
        logits=zero_shot_logits,
        labels=split_labels["test"],
        indices=split_indices["test"],
        class_names=class_names,
        method_name="zero_shot",
    )
    save_evaluation_artifacts(output_dir, "zero_shot", zero_shot_eval, class_names)

    print("Training few-shot linear probe...")
    classifier, history_rows, best_dev_metrics = train_linear_probe(
        train_features=split_features["few_shot_train"],
        train_labels=split_labels["few_shot_train"],
        dev_features=split_features["few_shot_dev"],
        dev_labels=split_labels["few_shot_dev"],
        config=config,
        class_names=class_names,
        device=device,
        output_dir=output_dir,
    )
    save_rows_csv(
        output_dir / "few_shot_training_history.csv",
        history_rows,
        fieldnames=[
            "epoch",
            "train_loss",
            "dev_loss",
            "dev_accuracy",
            "dev_macro_f1",
        ],
    )

    few_shot_logits = predict_linear_probe_logits(
        classifier=classifier,
        features=split_features["test"],
        batch_size=config.batch_size * 8,
        device=device,
    )
    few_shot_eval = evaluate_logits(
        logits=few_shot_logits,
        labels=split_labels["test"],
        indices=split_indices["test"],
        class_names=class_names,
        method_name="few_shot_linear_probe",
    )
    save_evaluation_artifacts(
        output_dir,
        "few_shot_linear_probe",
        few_shot_eval,
        class_names,
    )

    comparison_rows = [
        {"method": "zero_shot", **zero_shot_eval["metrics"]},
        {"method": "few_shot_linear_probe", **few_shot_eval["metrics"]},
    ]
    save_rows_csv(
        output_dir / "comparison.csv",
        comparison_rows,
        fieldnames=[
            "method",
            "accuracy",
            "balanced_accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "weighted_precision",
            "weighted_recall",
            "weighted_f1",
        ],
    )

    comparison_summary = {
        "class_names": class_names,
        "evaluation_split_name": "test",
        "best_dev_metrics": best_dev_metrics,
        "zero_shot": zero_shot_eval["metrics"],
        "few_shot_linear_probe": few_shot_eval["metrics"],
        "metric_deltas": {
            metric_name: few_shot_eval["metrics"][metric_name]
            - zero_shot_eval["metrics"][metric_name]
            for metric_name in zero_shot_eval["metrics"]
        },
    }
    save_json(output_dir / "comparison_summary.json", comparison_summary)

    final_summary = {
        "output_dir": str(output_dir),
        "dataset_summary": dataset_summary,
        "comparison_summary": comparison_summary,
    }
    save_json(output_dir / "run_summary.json", final_summary)

    print(f"Artifacts saved to {output_dir}")
    print(
        "Zero-shot accuracy: "
        f"{zero_shot_eval['metrics']['accuracy']:.4f} | "
        "Few-shot accuracy: "
        f"{few_shot_eval['metrics']['accuracy']:.4f}"
    )
    return final_summary


def main() -> None:
    config = parse_args()
    run_experiment(config)


if __name__ == "__main__":
    main()
