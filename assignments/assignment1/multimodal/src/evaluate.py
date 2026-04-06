from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix
from torch import nn
from transformers import AutoProcessor, CLIPModel

from common import (
    ensure_dir,
    humanize_class_name,
    load_config,
    load_rows_csv,
    resolve_device,
    resolve_run_dir,
    sanitize_model_id,
    save_json,
    save_rows_csv,
)
from infer import extract_single_image_embedding, load_coop_prompt_learner, load_linear_probe
from train import build_coop_text_features

matplotlib.use("Agg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a Food101 run, generate summary plots, confusion matrices, "
            "decode CoOp context vectors, and analyze failed predictions."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("assignments/assignment1/multimodal/configs/food101_clip.yaml"),
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Explicit training run directory. Defaults to the latest run.",
    )
    parser.add_argument("--top-failures", type=int, default=5)
    return parser.parse_args()


def compute_confusion(true_labels: list[int], predicted_labels: list[int], num_classes: int) -> np.ndarray:
    return confusion_matrix(true_labels, predicted_labels, labels=list(range(num_classes)))


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def plot_metric_bars(summary_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = summary_df.copy()
    plot_df["setting"] = plot_df.apply(
        lambda row: "zero_shot" if row["method"] == "zero_shot" else f"{row['method']}_{int(row['shots'])}",
        axis=1,
    )
    melted = plot_df.melt(
        id_vars=["setting"],
        value_vars=["accuracy", "macro_precision", "macro_f1"],
        var_name="metric",
        value_name="score",
    )
    plt.figure(figsize=(14, 6))
    sns.barplot(data=melted, x="setting", y="score", hue="metric")
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_confusion_matrix(matrix: np.ndarray, class_names: list[str], title: str, output_path: Path) -> None:
    labels = [humanize_class_name(name) for name in class_names]
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def decode_coop_context_tokens(
    clip_model: CLIPModel,
    processor: AutoProcessor,
    prompt_checkpoint: Path,
    class_names: list[str],
) -> list[dict[str, Any]]:
    prompt_learner = load_coop_prompt_learner(
        checkpoint_path=prompt_checkpoint,
        clip_model=clip_model,
        processor=processor,
        class_names=class_names,
    )
    token_embedding = clip_model.text_model.embeddings.token_embedding.weight.detach().cpu()
    tokenizer = processor.tokenizer
    context_tensor = prompt_learner.context.detach().cpu()
    if context_tensor.ndim == 2:
        context_tensor = context_tensor.unsqueeze(0).repeat(len(class_names), 1, 1)

    rows: list[dict[str, Any]] = []
    for class_index, class_name in enumerate(class_names):
        for token_index in range(context_tensor.shape[1]):
            context_vector = context_tensor[class_index, token_index]
            similarities = torch.nn.functional.cosine_similarity(
                context_vector.unsqueeze(0),
                token_embedding,
                dim=1,
            )
            top_values, top_indices = torch.topk(similarities, k=5)
            rows.append(
                {
                    "class_name": class_name,
                    "context_token_index": token_index,
                    "nearest_tokens": [
                        tokenizer.decode([int(token_id)]).strip() or f"<id:{int(token_id)}>"
                        for token_id in top_indices.tolist()
                    ],
                    "nearest_scores": [float(value) for value in top_values.tolist()],
                }
            )
    return rows


def build_failed_rows(
    predictions_path: Path,
    method: str,
    shots: int,
    image_path_by_record: dict[str, str],
) -> list[dict[str, Any]]:
    rows = load_rows_csv(predictions_path)
    failed_rows: list[dict[str, Any]] = []
    for row in rows:
        if row["true_label_id"] == row["predicted_label_id"]:
            continue
        failed_rows.append(
            {
                "method": method,
                "shots": shots,
                "record_id": row["record_id"],
                "true_label_id": int(row["true_label_id"]),
                "true_label_name": row["true_label_name"],
                "predicted_label_id": int(row["predicted_label_id"]),
                "predicted_label_name": row["predicted_label_name"],
                "confidence": float(row["confidence"]),
                "image_path": image_path_by_record[row["record_id"]],
            }
        )
    failed_rows.sort(key=lambda item: item["confidence"], reverse=True)
    return failed_rows


def save_saliency_map(
    image_path: Path,
    output_path: Path,
    logits_fn,
    target_index: int,
    processor: AutoProcessor,
    device: torch.device,
) -> None:
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    pixel_values.requires_grad_(True)

    logits = logits_fn(pixel_values)
    score = logits[0, target_index]
    score.backward()
    gradients = pixel_values.grad.detach().abs().max(dim=1)[0][0].cpu().numpy()
    gradients = (gradients - gradients.min()) / (gradients.max() - gradients.min() + 1e-8)

    image_np = np.array(image)
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(image_np)
    plt.axis("off")
    plt.title("Image")
    plt.subplot(1, 2, 2)
    plt.imshow(image_np)
    plt.imshow(gradients, cmap="jet", alpha=0.45)
    plt.axis("off")
    plt.title("Saliency")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def extract_clip_image_features(clip_model: CLIPModel, pixel_values: torch.Tensor) -> torch.Tensor:
    outputs = clip_model.get_image_features(pixel_values=pixel_values)
    if hasattr(outputs, "pooler_output"):
        image_features = outputs.pooler_output
    elif isinstance(outputs, (list, tuple)):
        image_features = outputs[0]
    else:
        image_features = outputs
    return image_features / image_features.norm(dim=-1, keepdim=True)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    model_id = config["model"]["id"]
    model_tag = sanitize_model_id(model_id)
    run_root = Path(config["paths"]["run_root"]) / model_tag
    run_dir = resolve_run_dir(run_root=run_root, run_dir=args.run_dir)
    eval_root = ensure_dir(run_dir / "evaluation")
    plot_root = ensure_dir(eval_root / "plots")
    confusion_root = ensure_dir(eval_root / "confusion_matrices")
    failure_root = ensure_dir(eval_root / "failed_cases")
    saliency_root = ensure_dir(eval_root / "saliency")

    device = resolve_device(config["training"].get("device"))
    processor = AutoProcessor.from_pretrained(model_id, use_fast=False)
    clip_model = CLIPModel.from_pretrained(model_id).to(device)
    clip_model.eval()

    class_names = list(config["dataset"]["selected_classes"])
    processed_root = Path(config["paths"]["processed_root"])
    manifests_root = processed_root / "manifests"
    test_manifest = load_rows_csv(manifests_root / "test.csv")
    image_path_by_record = {row["record_id"]: row["image_path"] for row in test_manifest}
    true_label_by_record = {row["record_id"]: int(row["selected_label_id"]) for row in test_manifest}

    summary_df = pd.read_csv(run_dir / "summary.csv")
    summary_df.to_csv(eval_root / "metrics_summary.csv", index=False)
    plot_metric_bars(summary_df, plot_root / "comparison_bar.png")

    zero_preds = load_rows_csv(run_dir / "zero_shot_predictions.csv")
    zero_true = [true_label_by_record[row["record_id"]] for row in zero_preds]
    zero_pred = [int(row["predicted_label_id"]) for row in zero_preds]
    zero_matrix = compute_confusion(zero_true, zero_pred, len(class_names))
    plot_confusion_matrix(
        zero_matrix,
        class_names,
        "Zero-shot Confusion Matrix",
        confusion_root / "zero_shot.png",
    )

    failure_rows: list[dict[str, Any]] = build_failed_rows(
        run_dir / "zero_shot_predictions.csv",
        method="zero_shot",
        shots=0,
        image_path_by_record=image_path_by_record,
    )

    text_payload = torch.load(
        Path(config["paths"]["embedding_root"]) / model_tag / "text_embeddings.pt",
        map_location="cpu",
    )

    for shots in [int(value) for value in config["few_shot"]["num_shots"]]:
        probe_path = run_dir / f"fewshot_{shots}" / "linear_probe" / "predictions.csv"
        if probe_path.exists():
            probe_preds = load_rows_csv(probe_path)
            probe_true = [true_label_by_record[row["record_id"]] for row in probe_preds]
            probe_pred = [int(row["predicted_label_id"]) for row in probe_preds]
            probe_matrix = compute_confusion(probe_true, probe_pred, len(class_names))
            plot_confusion_matrix(
                probe_matrix,
                class_names,
                f"Linear Probe Confusion Matrix ({shots} shots)",
                confusion_root / f"linear_probe_{shots}.png",
            )
            failure_rows.extend(
                build_failed_rows(
                    probe_path,
                    method="linear_probe",
                    shots=shots,
                    image_path_by_record=image_path_by_record,
                )
            )

        coop_path = run_dir / f"fewshot_{shots}" / "coop" / "predictions.csv"
        prompt_checkpoint = run_dir / f"fewshot_{shots}" / "coop" / "prompt_learner.pt"
        if coop_path.exists():
            coop_preds = load_rows_csv(coop_path)
            coop_true = [true_label_by_record[row["record_id"]] for row in coop_preds]
            coop_pred = [int(row["predicted_label_id"]) for row in coop_preds]
            coop_matrix = compute_confusion(coop_true, coop_pred, len(class_names))
            plot_confusion_matrix(
                coop_matrix,
                class_names,
                f"CoOp Confusion Matrix ({shots} shots)",
                confusion_root / f"coop_{shots}.png",
            )
            failure_rows.extend(
                build_failed_rows(
                    coop_path,
                    method="coop",
                    shots=shots,
                    image_path_by_record=image_path_by_record,
                )
            )
            decoded_rows = decode_coop_context_tokens(
                clip_model=clip_model,
                processor=processor,
                prompt_checkpoint=prompt_checkpoint,
                class_names=class_names,
            )
            save_json(
                eval_root / f"coop_{shots}_decoded_context.json",
                {"rows": decoded_rows},
            )

    failure_rows.sort(key=lambda item: item["confidence"], reverse=True)
    top_failures = failure_rows[: args.top_failures]
    save_rows_csv(
        failure_root / "top_failures.csv",
        top_failures,
        [
            "method",
            "shots",
            "record_id",
            "true_label_id",
            "true_label_name",
            "predicted_label_id",
            "predicted_label_name",
            "confidence",
            "image_path",
        ],
    )

    for index, failed in enumerate(top_failures):
        image_path = processed_root / failed["image_path"]
        output_path = saliency_root / (
            f"{index:02d}_{failed['method']}_shot{failed['shots']}_"
            f"true-{failed['true_label_name']}_pred-{failed['predicted_label_name']}.png"
        )

        if failed["method"] == "zero_shot":
            text_features = text_payload["class_embeddings"].to(device)
            logit_scale = float(text_payload["logit_scale"])

            def logits_fn(pixel_values: torch.Tensor) -> torch.Tensor:
                image_features = extract_clip_image_features(clip_model, pixel_values)
                return logit_scale * (image_features @ text_features.T)

        elif failed["method"] == "linear_probe":
            checkpoint = run_dir / f"fewshot_{failed['shots']}" / "linear_probe" / "linear_probe.pt"
            classifier = load_linear_probe(
                checkpoint_path=checkpoint,
                embedding_dim=int(text_payload["class_embeddings"].shape[1]),
                num_classes=len(class_names),
            ).to(device)

            def logits_fn(pixel_values: torch.Tensor) -> torch.Tensor:
                features = extract_clip_image_features(clip_model, pixel_values)
                return classifier(features)

        else:
            checkpoint = run_dir / f"fewshot_{failed['shots']}" / "coop" / "prompt_learner.pt"
            prompt_learner = load_coop_prompt_learner(
                checkpoint_path=checkpoint,
                clip_model=clip_model,
                processor=processor,
                class_names=class_names,
            ).to(device)
            prompt_learner.eval()

            def logits_fn(pixel_values: torch.Tensor) -> torch.Tensor:
                features = extract_clip_image_features(clip_model, pixel_values)
                text_features = build_coop_text_features(prompt_learner, clip_model, device=device)
                return clip_model.logit_scale.exp() * (features @ text_features.T)

        save_saliency_map(
            image_path=image_path,
            output_path=output_path,
            logits_fn=logits_fn,
            target_index=int(failed["predicted_label_id"]),
            processor=processor,
            device=device,
        )

    summary_payload = {
        "run_dir": str(run_dir),
        "num_methods": int(summary_df["method"].nunique()),
        "num_rows": int(len(summary_df)),
        "top_failure_count": len(top_failures),
        "class_names": class_names,
        "top_failure_distribution": dict(Counter(row["predicted_label_name"] for row in top_failures)),
    }
    save_json(eval_root / "summary.json", summary_payload)
    print(json.dumps(summary_payload, indent=2))


if __name__ == "__main__":
    main()
