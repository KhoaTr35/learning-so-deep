from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[3]
RUN_ROOT = ROOT / "assignments" / "assignment1" / "multimodal" / "notebooks" / "output" / "clip-vit-base-patch32"
REPORT_ROOT = ROOT / "assignments" / "assignment1" / "multimodal" / "artifacts" / "report_20260326_170105"
EDA_SUMMARY_PATH = ROOT / "docs" / "assignment-1" / "multimodal" / "assets" / "dataset-eda" / "eda_summary.json"
DOCS_ASSET_ROOT = ROOT / "docs" / "assignment-1" / "multimodal" / "assets" / "report-data"


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def latest_run_dir() -> Path:
    candidates = [path for path in RUN_ROOT.iterdir() if path.is_dir() and (path / "comparison_summary.json").exists()]
    if not candidates:
        raise FileNotFoundError(f"No multimodal run directories found under {RUN_ROOT}")
    return sorted(candidates)[-1]


def as_percent(value: float) -> float:
    return round(value * 100, 2)


def build_pipeline_data() -> Dict[str, Any]:
    run_dir = latest_run_dir()
    config = load_json(run_dir / "config.json")
    comparison = load_json(run_dir / "comparison_summary.json")
    dataset_summary = load_json(run_dir / "dataset_summary.json")
    eda_summary = load_json(EDA_SUMMARY_PATH)
    history_rows = load_csv_rows(run_dir / "few_shot_training_history.csv")
    metrics_rows = load_csv_rows(REPORT_ROOT / "metrics_summary.csv")

    few_shot_curve = [
        {
            "epoch": int(row["epoch"]),
            "train_loss": round(float(row["train_loss"]), 4),
            "dev_loss": round(float(row["dev_loss"]), 4),
            "dev_accuracy": as_percent(float(row["dev_accuracy"])),
            "dev_macro_f1": as_percent(float(row["dev_macro_f1"])),
        }
        for row in history_rows
    ]

    metrics_by_shot = sorted(
        [
            {
                "shots_per_class": int(row["shots_per_class"]),
                "method": row["method"],
                "run_timestamp": row["run_timestamp"],
                "accuracy": as_percent(float(row["accuracy"])),
                "precision": as_percent(float(row["precision"])),
                "f1_score": as_percent(float(row["f1_score"])),
                "balanced_accuracy": as_percent(float(row["balanced_accuracy"])),
            }
            for row in metrics_rows
        ],
        key=lambda row: (row["shots_per_class"], row["method"]),
    )

    zero = comparison["zero_shot"]
    few = comparison["few_shot_linear_probe"]

    pipeline_stages = [
        {
            "id": "input",
            "title": "Input X",
            "eyebrow": "Stage 01",
            "summary": "A single RGB food image enters the system, either from the test set or a user-uploaded example in the Streamlit demo.",
            "details": [
                "Images come from the selected 10-class Food101 subset.",
                "The same image can be evaluated by both zero-shot CLIP and the few-shot linear probe.",
                "Labels are only used for supervision during training and for metric computation during test-time evaluation.",
            ],
            "code": "image = Image.open(uploaded).convert(\"RGB\")\ninputs = processor(images=image, return_tensors=\"pt\")\npixel_values = inputs[\"pixel_values\"].to(device)",
        },
        {
            "id": "preprocess",
            "title": "Preprocessing",
            "eyebrow": "Stage 02",
            "summary": "The CLIP processor handles resizing, normalization, and tensor packaging so both branches share the same visual encoder input format.",
            "details": [
                "The Hugging Face AutoProcessor standardizes images for `clip-vit-base-patch32`.",
                "Prompt templates are also tokenized by the same processor in the zero-shot branch.",
                "Using one processor removes train-test preprocessing drift between methods.",
            ],
            "code": "processor = AutoProcessor.from_pretrained(model_id)\nimage_inputs = processor(images=batch[\"images\"], return_tensors=\"pt\")\ntext_inputs = processor(text=prompts, return_tensors=\"pt\", padding=True, truncation=True)",
        },
        {
            "id": "backbone",
            "title": "Backbone",
            "eyebrow": "Stage 03",
            "summary": "CLIP ViT-B/32 maps images and text into the same embedding space, enabling direct zero-shot matching and reusable image features for few-shot learning.",
            "details": [
                "Model ID: `openai/clip-vit-base-patch32`.",
                "Image embeddings are L2-normalized before classification.",
                "The text tower averages five prompt templates per class to stabilize zero-shot predictions.",
            ],
            "code": "model = CLIPModel.from_pretrained(model_id).to(device)\nimage_features = model.get_image_features(pixel_values=pixel_values)\nimage_features = image_features / image_features.norm(dim=-1, keepdim=True)",
        },
        {
            "id": "head",
            "title": "Classifier Head",
            "eyebrow": "Stage 04",
            "summary": "This is where the two methods diverge: zero-shot uses text embeddings as class prototypes, while few-shot trains a linear probe on top of frozen image embeddings.",
            "details": [
                "Zero-shot head: cosine-style similarity with CLIP logit scaling against prompt-derived text features.",
                "Few-shot head: linear layer trained for 20 epochs with learning rate 1e-3 and weight decay 1e-4.",
                "Best saved 128-shot dev checkpoint reached 97.00% dev accuracy.",
            ],
            "code": "zero_shot_logits = logit_scale * (image_features @ text_features.T)\nprobe = nn.Linear(feature_dim, num_classes)\nlogits = probe(image_features)",
        },
        {
            "id": "results",
            "title": "Classification Results",
            "eyebrow": "Stage 05",
            "summary": "Probabilities, confusion matrices, and metric summaries are produced on the held-out test split for both methods.",
            "details": [
                f"Zero-shot test accuracy: {as_percent(zero['accuracy']):.2f}%.",
                f"Few-shot linear probe test accuracy: {as_percent(few['accuracy']):.2f}% at 128 shots.",
                "Predictions, classification reports, and confusion matrices are exported for downstream error analysis.",
            ],
            "code": "probs = torch.softmax(logits, dim=-1)\nvalues, indices = torch.topk(probs, k=min(top_k, probs.numel()))\nsave_evaluation_artifacts(output_dir, method_name, evaluation, class_names)",
        },
    ]

    return {
        "meta": {
            "model_id": config["model_id"],
            "dataset_name": config["dataset_name"],
            "run_timestamp": run_dir.name,
            "classes": [name.replace("_", " ").title() for name in config["class_names"]],
            "prompt_templates": config["prompt_templates"],
            "links": {
                "backbone": "./model-backbone.html",
                "methodology": "./methodology.html",
                "results": "./evaluation-results.html",
                "eda": "./dataset-eda.html",
            },
        },
        "dataset": {
            "selected_subset_total_count": eda_summary["dataset"]["selected_subset_total_count"],
            "active_experiment_total_count": eda_summary["dataset"]["active_experiment_total_count"],
            "few_shot_train_size": dataset_summary["few_shot_train_size"],
            "few_shot_dev_size": dataset_summary["few_shot_dev_size"],
            "test_size": dataset_summary.get("held_out_test_size", dataset_summary.get("held_out_validation_size")),
            "shots_per_class": config["shots_per_class"],
            "dev_per_class": config["val_per_class"],
        },
        "pipeline": pipeline_stages,
        "methods": {
            "zero_shot": {
                "title": "Zero-shot CLIP",
                "tagline": "No gradient updates on classifier weights. Class text prompts act as the head.",
                "strengths": [
                    "Strongest saved test performance in the current runs.",
                    "No task-specific head training required.",
                    "Naturally supports open-vocabulary prompt experimentation.",
                ],
                "limitations": [
                    "Sensitive to prompt wording and prompt-template coverage.",
                    "Performance ceiling is tied to pretrained semantic alignment.",
                ],
                "metrics": {
                    "accuracy": as_percent(zero["accuracy"]),
                    "balanced_accuracy": as_percent(zero["balanced_accuracy"]),
                    "macro_f1": as_percent(zero["macro_f1"]),
                },
            },
            "few_shot": {
                "title": "Few-shot Linear Probe",
                "tagline": "Frozen CLIP encoder with a learned linear classifier over image embeddings.",
                "strengths": [
                    "Adapts embeddings to the 10 Food101 classes.",
                    "Improves rapidly between 8 and 64 shots per class.",
                    "Training remains lightweight because only the probe is optimized.",
                ],
                "limitations": [
                    "Still underperforms the saved zero-shot baseline in these runs.",
                    "Requires labeled support examples and dev selection.",
                ],
                "metrics": {
                    "accuracy": as_percent(few["accuracy"]),
                    "balanced_accuracy": as_percent(few["balanced_accuracy"]),
                    "macro_f1": as_percent(few["macro_f1"]),
                    "best_dev_accuracy": as_percent(comparison["best_dev_metrics"]["accuracy"]),
                },
            },
        },
        "few_shot_curve": few_shot_curve,
        "metrics_by_shot": metrics_by_shot,
        "assets": {
            "comparison_bar": "comparison_bar.png",
            "training_curves": "training_curves.png",
            "zero_shot_confusion": "zero_shot_confusion.png",
            "few_shot_confusion": "few_shot_confusion.png",
        },
    }


def export_assets() -> None:
    DOCS_ASSET_ROOT.mkdir(parents=True, exist_ok=True)
    payload = build_pipeline_data()
    (DOCS_ASSET_ROOT / "multimodal_report_data.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    run_dir = latest_run_dir()
    copy_map = {
        run_dir / "comparison_bar.png": DOCS_ASSET_ROOT / "comparison_bar.png",
        run_dir / "training_curves.png": DOCS_ASSET_ROOT / "training_curves.png",
        run_dir / "zero_shot" / "confusion_matrix.png": DOCS_ASSET_ROOT / "zero_shot_confusion.png",
        run_dir / "few_shot_linear_probe" / "confusion_matrix.png": DOCS_ASSET_ROOT / "few_shot_confusion.png",
    }
    for src, dest in copy_map.items():
        shutil.copy2(src, dest)


def main() -> None:
    export_assets()
    print(f"Exported multimodal report data to {DOCS_ASSET_ROOT}")


if __name__ == "__main__":
    main()
