import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from PIL import Image
from transformers import AutoProcessor, CLIPModel

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT_DIR / "notebooks" / "output"
DEFAULT_ARTIFACT_ROOT = ROOT_DIR / "artifacts"
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"
TARGET_FEW_SHOT_SETTINGS = [8, 16, 32, 64, 128]
METHODS = ("zero_shot", "few_shot_linear_probe")


@dataclass
class RunRecord:
    run_dir: Path
    timestamp: str
    shots_per_class: int
    model_name: str
    config: Dict[str, Any]
    comparison: pd.DataFrame


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _humanize_class_name(name: str) -> str:
    return str(name).replace("_", " ")


def _resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_existing_path(path: Path) -> Path:
    candidates: List[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend(
            [
                path,
                Path.cwd() / path,
                NOTEBOOKS_DIR / path,
                ROOT_DIR / path,
            ]
        )

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def _looks_like_run_dir(path: Path) -> bool:
    return path.is_dir() and (path / "config.json").exists() and (path / "comparison.csv").exists()


def _iter_run_dirs(output_root: Path) -> List[Tuple[Path, Path]]:
    resolved_root = _resolve_existing_path(output_root)
    if not resolved_root.exists():
        raise FileNotFoundError(f"Output path does not exist: {resolved_root}")

    run_pairs: List[Tuple[Path, Path]] = []
    if _looks_like_run_dir(resolved_root):
        return [(resolved_root.parent, resolved_root)]

    direct_run_dirs = [child for child in sorted(resolved_root.iterdir()) if _looks_like_run_dir(child)]
    if direct_run_dirs:
        return [(resolved_root, run_dir) for run_dir in direct_run_dirs]

    for model_dir in sorted([d for d in resolved_root.iterdir() if d.is_dir()]):
        for run_dir in sorted([d for d in model_dir.iterdir() if _looks_like_run_dir(d)]):
            run_pairs.append((model_dir, run_dir))
    return run_pairs


def _collect_runs(output_root: Path) -> List[RunRecord]:
    runs: List[RunRecord] = []
    for model_dir, run_dir in _iter_run_dirs(output_root):
        config_path = run_dir / "config.json"
        comparison_path = run_dir / "comparison.csv"
        config = _load_json(config_path)
        comparison = pd.read_csv(comparison_path)
        runs.append(
            RunRecord(
                run_dir=run_dir,
                timestamp=run_dir.name,
                shots_per_class=int(config.get("shots_per_class", -1)),
                model_name=model_dir.name,
                config=config,
                comparison=comparison,
            )
        )
    if not runs:
        resolved_root = _resolve_existing_path(output_root)
        raise FileNotFoundError(
            "No valid runs found in "
            f"{resolved_root}. Expected either a run directory containing "
            "`config.json` and `comparison.csv`, a model directory containing timestamped "
            "run folders, or the parent notebooks/output root."
        )
    return runs


def _select_latest_runs_by_shot(runs: Sequence[RunRecord]) -> Dict[int, RunRecord]:
    latest: Dict[int, RunRecord] = {}
    for run in runs:
        shots = run.shots_per_class
        prev = latest.get(shots)
        if prev is None or run.timestamp > prev.timestamp:
            latest[shots] = run
    return latest


def _metric_from_run(run: RunRecord, method: str, metric: str) -> float:
    row = run.comparison[run.comparison["method"] == method]
    if row.empty or metric not in row.columns:
        return float("nan")
    return float(row.iloc[0][metric])


def _create_run_artifact_dir(artifact_root: Path, run_name: Optional[str]) -> Path:
    if run_name is None or not str(run_name).strip():
        run_name = datetime.now().strftime("results_%Y%m%d_%H%M%S")
    out = artifact_root / run_name
    out.mkdir(parents=True, exist_ok=True)
    for sub in ("plots", "predictions", "saliency"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    return out


def _build_metric_summary(latest_by_shot: Dict[int, RunRecord]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for shots in sorted(latest_by_shot):
        run = latest_by_shot[shots]
        for method in METHODS:
            rows.append(
                {
                    "shots_per_class": shots,
                    "method": method,
                    "run_timestamp": run.timestamp,
                    "accuracy": _metric_from_run(run, method, "accuracy"),
                    "precision": _metric_from_run(run, method, "macro_precision"),
                    "f1_score": _metric_from_run(run, method, "macro_f1"),
                    "balanced_accuracy": _metric_from_run(run, method, "balanced_accuracy"),
                }
            )
    return pd.DataFrame(rows).sort_values(["shots_per_class", "method"])


def _plot_metrics_bar(latest_by_shot: Dict[int, RunRecord], out_path: Path) -> None:
    zero_shot_source = latest_by_shot[sorted(latest_by_shot.keys())[-1]]
    settings_labels = ["zero_shot"] + [f"few_shot_{k}" for k in TARGET_FEW_SHOT_SETTINGS]

    metric_values: Dict[str, List[float]] = {"accuracy": [], "precision": [], "f1_score": []}
    for metric, src_metric in (
        ("accuracy", "accuracy"),
        ("precision", "macro_precision"),
        ("f1_score", "macro_f1"),
    ):
        metric_values[metric].append(_metric_from_run(zero_shot_source, "zero_shot", src_metric))
        for k in TARGET_FEW_SHOT_SETTINGS:
            run = latest_by_shot.get(k)
            if run is None:
                metric_values[metric].append(np.nan)
            else:
                metric_values[metric].append(
                    _metric_from_run(run, "few_shot_linear_probe", src_metric)
                )

    x = np.arange(len(settings_labels))
    width = 0.24
    fig, ax = plt.subplots(figsize=(13, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for idx, metric in enumerate(["accuracy", "precision", "f1_score"]):
        vals = metric_values[metric]
        bars = ax.bar(x + (idx - 1) * width, vals, width=width, label=metric, color=colors[idx])
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.004, f"{h:.3f}", ha="center", fontsize=8)

    ax.set_title("Zero-shot vs Few-shot Settings (Accuracy / Precision / F1)")
    ax.set_ylabel("Score")
    ax.set_xlabel("Method setting")
    ax.set_xticks(x)
    ax.set_xticklabels(settings_labels, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _load_confusion_matrix(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    if "true_label" in df.columns:
        df = df.set_index("true_label")
    return df


def _plot_confusion_comparison(latest_by_shot: Dict[int, RunRecord], out_path: Path) -> None:
    shots_available = [k for k in TARGET_FEW_SHOT_SETTINGS if k in latest_by_shot]
    if not shots_available:
        return
    fig, axes = plt.subplots(
        nrows=len(shots_available),
        ncols=2,
        figsize=(16, 4.3 * len(shots_available)),
    )
    if len(shots_available) == 1:
        axes = np.array([axes])

    for row_idx, shots in enumerate(shots_available):
        run = latest_by_shot[shots]
        zs_cm = _load_confusion_matrix(run.run_dir / "zero_shot" / "confusion_matrix.csv")
        fs_cm = _load_confusion_matrix(
            run.run_dir / "few_shot_linear_probe" / "confusion_matrix.csv"
        )
        for col_idx, (title, cm) in enumerate(
            [
                (f"Zero-shot (shot={shots})", zs_cm),
                (f"Few-shot linear probe (shot={shots})", fs_cm),
            ]
        ):
            ax = axes[row_idx, col_idx]
            if cm is None:
                ax.set_title(f"{title}\nMissing confusion_matrix.csv")
                ax.axis("off")
                continue
            sns.heatmap(cm, cmap="YlOrRd", cbar=True, ax=ax)
            ax.set_title(title)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_linear_probe_loss(latest_by_shot: Dict[int, RunRecord], out_path: Path) -> None:
    shots_available = [k for k in TARGET_FEW_SHOT_SETTINGS if k in latest_by_shot]
    if not shots_available:
        return
    cols = 3
    rows = int(np.ceil(len(shots_available) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4.2 * rows))
    axes = np.atleast_1d(axes).flatten()

    for idx, shots in enumerate(shots_available):
        ax = axes[idx]
        run = latest_by_shot[shots]
        history_path = run.run_dir / "few_shot_training_history.csv"
        if not history_path.exists():
            ax.set_title(f"shot={shots} (history missing)")
            ax.axis("off")
            continue
        history = pd.read_csv(history_path)
        if "epoch" not in history.columns or "train_loss" not in history.columns:
            ax.set_title(f"shot={shots} (invalid history)")
            ax.axis("off")
            continue
        ax.plot(history["epoch"], history["train_loss"], marker="o", label="train_loss", color="#1f77b4")
        if "dev_loss" in history.columns:
            ax.plot(history["epoch"], history["dev_loss"], marker="o", label="dev_loss", color="#d62728")
        ax.set_title(f"Linear probe loss (shot={shots})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(alpha=0.25)

    for idx in range(len(shots_available), len(axes)):
        axes[idx].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _pick_examples(preds: pd.DataFrame, examples_per_method: int) -> pd.DataFrame:
    if preds.empty:
        return preds
    preds = preds.copy()
    preds["correct"] = preds["true_label_id"] == preds["predicted_label_id"]
    wrong = preds[preds["correct"] == False].sort_values("confidence")
    right = preds[preds["correct"] == True].sort_values("confidence", ascending=False)
    half = max(examples_per_method // 2, 1)
    selected = pd.concat([wrong.head(half), right.head(examples_per_method - half)], axis=0)
    selected = selected.drop_duplicates(subset=["sample_index"]).head(examples_per_method)
    if selected.empty:
        selected = preds.head(examples_per_method)
    return selected


def _extract_evaluation_dataset(dataset: Any) -> Any:
    if dataset is None:
        raise ValueError("dataset is required for prediction/saliency visualization.")
    if isinstance(dataset, dict):
        if "test" in dataset:
            return dataset["test"]
        if "validation" in dataset:
            return dataset["validation"]
        raise ValueError("If dataset is dict-like, it must contain a 'test' or 'validation' key.")
    return dataset


def _image_from_dataset(evaluation_dataset: Any, sample_index: int) -> Image.Image:
    sample = evaluation_dataset[sample_index]
    if isinstance(sample, dict):
        image = sample.get("image")
    else:
        image = sample["image"]
    if not isinstance(image, Image.Image):
        raise TypeError("Expected a PIL image in dataset sample['image'].")
    return image.convert("RGB")


def _choose_prompt(
    image: Image.Image,
    predicted_label_name: str,
    prompt_templates: Sequence[str],
    model: CLIPModel,
    processor: AutoProcessor,
    device: torch.device,
) -> Tuple[str, List[str]]:
    prompts = [tpl.format(_humanize_class_name(predicted_label_name)) for tpl in prompt_templates]
    with torch.no_grad():
        image_inputs = processor(images=image, return_tensors="pt")
        pixel_values = image_inputs["pixel_values"].to(device)
        outputs = model.get_image_features(pixel_values=pixel_values)
        if hasattr(outputs, "pooler_output"):
            image_features = outputs.pooler_output
        elif isinstance(outputs, (list, tuple)):
            image_features = outputs[0]
        else:
            image_features = outputs
        
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        text_inputs = processor(text=prompts, return_tensors="pt", padding=True, truncation=True)
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
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

        sims = (image_features @ text_features.T).squeeze(0).cpu().numpy()
    chosen = prompts[int(np.argmax(sims))]
    return chosen, prompts


def _compute_saliency(
    image: Image.Image,
    prompt: str,
    model: CLIPModel,
    processor: AutoProcessor,
    device: torch.device,
) -> np.ndarray:
    # Ensure saliency works even if caller previously used inference_mode().
    with torch.inference_mode(False), torch.enable_grad():
        text_inputs = processor(text=[prompt], return_tensors="pt", padding=True, truncation=True)
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        with torch.no_grad():
            text_outputs = model.get_text_features(**text_inputs)
            if hasattr(text_outputs, "pooler_output"):
                text_features = text_outputs.pooler_output
            elif isinstance(text_outputs, (list, tuple)):
                text_features = text_outputs[0]
            else:
                text_features = text_outputs
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_features = text_features.detach().clone()

        image_inputs = processor(images=image, return_tensors="pt")
        pixel_values = image_inputs["pixel_values"].to(device).detach().clone()
        pixel_values.requires_grad_(True)

        outputs = model.get_image_features(pixel_values=pixel_values)
        if hasattr(outputs, "pooler_output"):
            image_features = outputs.pooler_output
        elif isinstance(outputs, (list, tuple)):
            image_features = outputs[0]
        else:
            image_features = outputs

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        image_features = image_features.clone()
        score = (image_features * text_features).sum()
        grad_tensor = torch.autograd.grad(score, pixel_values, retain_graph=False, create_graph=False)[0]

    grad = grad_tensor.detach().abs().mean(dim=1).squeeze(0).cpu().numpy()
    grad -= grad.min()
    grad /= max(float(grad.max()), 1e-8)
    return grad


def _save_prediction_panel(
    image: Image.Image,
    row: pd.Series,
    prompts: Sequence[str],
    chosen_prompt: str,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.2, 1.8]})
    axes[0].imshow(image)
    axes[0].axis("off")
    axes[0].set_title("Input image")

    prompt_text = "\n".join([f"- {prompt}" for prompt in prompts])
    detail = (
        f"Method: {row['method']}\n"
        f"Sample index: {int(row['sample_index'])}\n"
        f"True label: {_humanize_class_name(row['true_label_name'])}\n"
        f"Predicted: {_humanize_class_name(row['predicted_label_name'])}\n"
        f"Confidence: {float(row['confidence']):.4f}\n\n"
        f"Chosen prompt:\n{chosen_prompt}\n\n"
        f"Given prompts:\n{prompt_text}"
    )
    axes[1].text(0.0, 1.0, detail, va="top", ha="left", fontsize=10, wrap=True)
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _save_saliency_panel(
    image: Image.Image,
    saliency: np.ndarray,
    row: pd.Series,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].imshow(image)
    axes[0].axis("off")
    axes[0].set_title("Original image")

    sal = Image.fromarray((saliency * 255).astype(np.uint8)).resize(image.size)
    axes[1].imshow(image)
    axes[1].imshow(np.asarray(sal), cmap="turbo", alpha=0.45)
    axes[1].axis("off")
    axes[1].set_title(f"Prediction-focused pixels\n{_humanize_class_name(row['predicted_label_name'])}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def build_prediction_and_saliency(
    latest_by_shot: Dict[int, RunRecord],
    dataset: Any,
    output_dir: Path,
    examples_per_method: int = 2,
    model: Optional[CLIPModel] = None,
    processor: Optional[AutoProcessor] = None,
) -> None:
    evaluation_dataset = _extract_evaluation_dataset(dataset)
    device = _resolve_device()
    model_cache: Dict[str, Tuple[CLIPModel, AutoProcessor]] = {}
    index_rows: List[Dict[str, Any]] = []

    for shots in TARGET_FEW_SHOT_SETTINGS:
        run = latest_by_shot.get(shots)
        if run is None:
            continue
        model_id = run.config.get("model_id", "openai/clip-vit-base-patch32")
        if model is not None and processor is not None:
            clip_model = model.to(device)
            clip_model.eval()
            clip_processor = processor
        else:
            if model_id not in model_cache:
                loaded_model = CLIPModel.from_pretrained(model_id).to(device)
                loaded_model.eval()
                loaded_processor = AutoProcessor.from_pretrained(model_id)
                model_cache[model_id] = (loaded_model, loaded_processor)
            clip_model, clip_processor = model_cache[model_id]

        prompt_templates = run.config.get("prompt_templates", ["a photo of {}."])
        for method in METHODS:
            pred_path = run.run_dir / method / "predictions.csv"
            if not pred_path.exists():
                continue
            preds = pd.read_csv(pred_path)
            if preds.empty:
                continue
            examples = _pick_examples(preds, examples_per_method=examples_per_method)

            for _, row in examples.iterrows():
                sample_index = int(row["sample_index"])
                if sample_index >= len(evaluation_dataset):
                    continue
                image = _image_from_dataset(evaluation_dataset, sample_index)
                chosen_prompt, prompts = _choose_prompt(
                    image=image,
                    predicted_label_name=row["predicted_label_name"],
                    prompt_templates=prompt_templates,
                    model=clip_model,
                    processor=clip_processor,
                    device=device,
                )
                saliency = _compute_saliency(
                    image=image,
                    prompt=chosen_prompt,
                    model=clip_model,
                    processor=clip_processor,
                    device=device,
                )

                base = (
                    f"shot{shots}_{method}_idx{sample_index}_"
                    f"true-{row['true_label_name']}_pred-{row['predicted_label_name']}"
                )
                pred_out = output_dir / "predictions" / f"{base}.png"
                sal_out = output_dir / "saliency" / f"{base}_saliency.png"
                _save_prediction_panel(image, row, prompts, chosen_prompt, pred_out)
                _save_saliency_panel(image, saliency, row, sal_out)

                index_rows.append(
                    {
                        "shots_per_class": shots,
                        "method": method,
                        "sample_index": sample_index,
                        "true_label_name": row["true_label_name"],
                        "predicted_label_name": row["predicted_label_name"],
                        "confidence": float(row["confidence"]),
                        "chosen_prompt": chosen_prompt,
                        "prediction_path": str(pred_out),
                        "saliency_path": str(sal_out),
                    }
                )

    if index_rows:
        pd.DataFrame(index_rows).to_csv(
            output_dir / "prediction_saliency_index.csv",
            index=False,
        )


def generate_core_reports(
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    run_name: Optional[str] = None,
) -> Tuple[Path, Dict[int, RunRecord], pd.DataFrame]:
    runs = _collect_runs(output_root)
    latest_by_shot = _select_latest_runs_by_shot(runs)
    output_dir = _create_run_artifact_dir(artifact_root, run_name=run_name)

    metrics_df = _build_metric_summary(latest_by_shot)
    metrics_df.to_csv(output_dir / "metrics_summary.csv", index=False)

    _plot_metrics_bar(latest_by_shot, output_dir / "plots" / "bar_accuracy_precision_f1.png")
    _plot_confusion_comparison(
        latest_by_shot,
        output_dir / "plots" / "confusion_matrix_zero_vs_few_all_settings.png",
    )
    _plot_linear_probe_loss(
        latest_by_shot,
        output_dir / "plots" / "linear_probe_loss_all_settings.png",
    )
    return output_dir, latest_by_shot, metrics_df


def run_from_notebook(
    dataset: Any,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    run_name: Optional[str] = None,
    examples_per_method: int = 2,
    model: Optional[CLIPModel] = None,
    processor: Optional[AutoProcessor] = None,
) -> Path:
    output_dir, latest_by_shot, _ = generate_core_reports(
        output_root=output_root,
        artifact_root=artifact_root,
        run_name=run_name,
    )
    build_prediction_and_saliency(
        latest_by_shot=latest_by_shot,
        dataset=dataset,
        output_dir=output_dir,
        examples_per_method=examples_per_method,
        model=model,
        processor=processor,
    )
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate multimodal comparison artifacts. Use run_from_notebook(...) "
            "for prediction/saliency panels with the notebook dataset."
        )
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--run-name", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir, _, metrics_df = generate_core_reports(
        output_root=args.output_root,
        artifact_root=args.artifact_root,
        run_name=args.run_name,
    )
    print("Core report artifacts saved.")
    print(f"- Output folder: {output_dir}")
    print(f"- Metrics rows: {len(metrics_df)}")
    print(
        "Notebook step required for prediction/saliency panels:\n"
        "run_from_notebook(dataset=..., model=MODEL, processor=PROCESSOR, ...)"
    )


if __name__ == "__main__":
    main()
