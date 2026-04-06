from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd
import streamlit as st
import yaml
from PIL import Image

try:
    import torch
    from torch import nn
except ModuleNotFoundError:
    torch = None
    nn = None

from transformers import AutoProcessor, CLIPModel

try:
    from torchvision import models, transforms
except ModuleNotFoundError:
    models = None
    transforms = None


REPO_ROOT = Path(__file__).resolve().parents[2]
ASSIGNMENT_ROOT = Path(__file__).resolve().parent
IMAGE_ROOT = ASSIGNMENT_ROOT / "image"
TEXT_ROOT = ASSIGNMENT_ROOT / "text"
MULTIMODAL_ROOT = ASSIGNMENT_ROOT / "multimodal"

IMAGE_RUNS_ROOT = ASSIGNMENT_ROOT / "run" / "runs_caltech256"
PRIMARY_RUNS_ROOT = ASSIGNMENT_ROOT / "run" / "runs_food101_clip"
FALLBACK_RUNS_ROOT = MULTIMODAL_ROOT / "artifacts" / "runs_food101_clip"
DEFAULT_PROMPT_TEMPLATES = ("a photo of {}.",)


@dataclass(frozen=True)
class RunInfo:
    label: str
    model_dir: Path
    run_dir: Path
    model_id: str
    class_names: Tuple[str, ...]
    prompt_templates: Tuple[str, ...]
    shots_available: Tuple[int, ...]


def _humanize(name: str) -> str:
    return name.replace("_", " ").replace("-", " ")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _load_image(path: Path) -> Image.Image | None:
    if not path.exists():
        return None
    return Image.open(path)


def _display_image_if_exists(path: Path, caption: str, use_container_width: bool = True) -> None:
    image = _load_image(path)
    if image is not None:
        st.image(image, caption=caption, use_container_width=use_container_width)


def _image_transform():
    if transforms is None:
        raise ModuleNotFoundError("torchvision")
    return transforms.Compose(
        [
            transforms.Lambda(lambda image: image.convert("RGB")),
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def _list_prediction_examples(folder: Path, limit: int = 4) -> List[Path]:
    if not folder.exists():
        return []
    return sorted(
        [path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}]
    )[:limit]


def _resolve_device_label() -> str:
    if torch is None:
        return "unavailable"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_device() -> torch.device:
    return torch.device(_resolve_device_label())


def _extract_tensor(outputs: object) -> torch.Tensor:
    if hasattr(outputs, "pooler_output"):
        return outputs.pooler_output
    if isinstance(outputs, (list, tuple)):
        return outputs[0]
    return outputs  # type: ignore[return-value]


def _load_run_metadata(run_dir: Path, model_dir: Path) -> RunInfo | None:
    run_config_path = run_dir / "run_config.json"
    if not run_config_path.exists():
        return None

    run_config = _read_json(run_config_path)
    model_id = str(run_config.get("model_id", model_dir.name.replace("_", "/")))

    config_rel = run_config.get("config_path")
    config: dict = {}
    if config_rel:
        config_path = REPO_ROOT / str(config_rel)
        if config_path.exists():
            config = _read_yaml(config_path)

    eval_summary_path = run_dir / "evaluation" / "summary.json"
    eval_summary = _read_json(eval_summary_path) if eval_summary_path.exists() else {}

    class_names = tuple(
        eval_summary.get("class_names")
        or config.get("dataset", {}).get("selected_classes", [])
    )
    prompt_templates = tuple(
        config.get("prompts", {}).get("templates", DEFAULT_PROMPT_TEMPLATES)
    )

    shot_dirs: List[int] = []
    for candidate in sorted(run_dir.glob("fewshot_*")):
        if not candidate.is_dir():
            continue
        checkpoint_path = candidate / "linear_probe" / "linear_probe.pt"
        if checkpoint_path.exists():
            try:
                shot_dirs.append(int(candidate.name.split("_", maxsplit=1)[1]))
            except (IndexError, ValueError):
                continue

    if not class_names:
        return None

    return RunInfo(
        label=f"{model_dir.name} / {run_dir.name}",
        model_dir=model_dir,
        run_dir=run_dir,
        model_id=model_id,
        class_names=class_names,
        prompt_templates=prompt_templates,
        shots_available=tuple(shot_dirs),
    )


def _default_runs_root() -> Path:
    if PRIMARY_RUNS_ROOT.exists():
        return PRIMARY_RUNS_ROOT
    return FALLBACK_RUNS_ROOT


def _resolve_run_asset(run: RunInfo, relative_path: str) -> Path:
    primary = run.run_dir / relative_path
    if primary.exists():
        return primary

    try:
        suffix = run.run_dir.relative_to(PRIMARY_RUNS_ROOT)
    except ValueError:
        return primary

    fallback = FALLBACK_RUNS_ROOT / suffix / relative_path
    if fallback.exists():
        return fallback
    return primary


def _discover_runs(runs_root: Path) -> Dict[str, RunInfo]:
    runs: Dict[str, RunInfo] = {}
    if not runs_root.exists():
        return runs

    for model_dir in sorted([path for path in runs_root.iterdir() if path.is_dir()]):
        for run_dir in sorted([path for path in model_dir.iterdir() if path.is_dir()], reverse=True):
            run_info = _load_run_metadata(run_dir, model_dir)
            if run_info is not None:
                runs[run_info.label] = run_info

    return runs


@st.cache_data(show_spinner=False)
def _load_csv(path_str: str) -> pd.DataFrame:
    return pd.read_csv(path_str)


@st.cache_data(show_spinner=False)
def _load_json_data(path_str: str) -> dict:
    return _read_json(Path(path_str))


@st.cache_data(show_spinner=False)
def _load_categories(path_str: str) -> List[str]:
    return Path(path_str).read_text(encoding="utf-8").splitlines()


@st.cache_resource(show_spinner=False)
def _load_clip(model_id: str, device_str: str) -> Tuple[CLIPModel, AutoProcessor]:
    if torch is None:
        raise ModuleNotFoundError("torch")
    device = torch.device(device_str)
    processor = AutoProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).to(device)
    model.eval()
    return model, processor


@st.cache_resource(show_spinner=False)
def _load_resnet_classifier(checkpoint_path: str, device_str: str):
    if torch is None or nn is None or models is None:
        raise ModuleNotFoundError("torchvision")
    device = torch.device(device_str)
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 257)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


@st.cache_resource(show_spinner=False)
def _load_vit_classifier(checkpoint_path: str, device_str: str):
    if torch is None or nn is None or models is None:
        raise ModuleNotFoundError("torchvision")
    device = torch.device(device_str)
    model = models.vit_b_16(weights=None)
    num_ftrs = model.heads.head.in_features
    model.heads.head = nn.Linear(num_ftrs, 257)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


@st.cache_resource(show_spinner=False)
def _load_linear_probe(checkpoint_path: str, device_str: str) -> Tuple[nn.Linear, dict]:
    if torch is None or nn is None:
        raise ModuleNotFoundError("torch")
    device = torch.device(device_str)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["state_dict"]
    weight = state_dict["weight"]
    num_classes, feature_dim = weight.shape
    probe = nn.Linear(feature_dim, num_classes)
    probe.load_state_dict(state_dict)
    probe = probe.to(device)
    probe.eval()
    return probe, checkpoint


@st.cache_resource(show_spinner=False)
def _build_zero_shot_text_features(
    model_id: str,
    class_names: Tuple[str, ...],
    prompt_templates: Tuple[str, ...],
    device_str: str,
) -> torch.Tensor:
    if torch is None:
        raise ModuleNotFoundError("torch")
    device = torch.device(device_str)
    model, processor = _load_clip(model_id, device_str)
    text_feature_batches: List[torch.Tensor] = []

    with torch.inference_mode():
        for class_name in class_names:
            prompts = [template.format(_humanize(class_name)) for template in prompt_templates]
            text_inputs = processor(
                text=prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            text_inputs = {key: value.to(device) for key, value in text_inputs.items()}
            text_outputs = model.get_text_features(**text_inputs)
            text_features = _extract_tensor(text_outputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            averaged = text_features.mean(dim=0)
            averaged = averaged / averaged.norm()
            text_feature_batches.append(averaged)

    return torch.stack(text_feature_batches, dim=0).to(device)


def _predict_linear_probe(
    image: Image.Image,
    run: RunInfo,
    shots: int,
    top_k: int,
    device: torch.device,
) -> List[Tuple[str, float]]:
    if torch is None:
        raise ModuleNotFoundError("torch")
    checkpoint_path = run.run_dir / f"fewshot_{shots}" / "linear_probe" / "linear_probe.pt"
    model, processor = _load_clip(run.model_id, str(device))
    probe, checkpoint = _load_linear_probe(str(checkpoint_path), str(device))
    class_names = tuple(checkpoint.get("class_names", run.class_names))

    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    with torch.inference_mode():
        image_outputs = model.get_image_features(pixel_values=pixel_values)
        image_features = _extract_tensor(image_outputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = probe(image_features)
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        values, indices = torch.topk(probs, k=min(top_k, probs.numel()))

    return [(_humanize(class_names[index]), float(value)) for value, index in zip(values.tolist(), indices.tolist())]


def _predict_zero_shot(
    image: Image.Image,
    run: RunInfo,
    top_k: int,
    device: torch.device,
) -> List[Tuple[str, float]]:
    if torch is None:
        raise ModuleNotFoundError("torch")
    model, processor = _load_clip(run.model_id, str(device))
    text_features = _build_zero_shot_text_features(
        run.model_id,
        run.class_names,
        run.prompt_templates,
        str(device),
    )
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    with torch.inference_mode():
        image_outputs = model.get_image_features(pixel_values=pixel_values)
        image_features = _extract_tensor(image_outputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = model.logit_scale.exp()
        logits = logit_scale * (image_features @ text_features.T)
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        values, indices = torch.topk(probs, k=min(top_k, probs.numel()))

    return [(_humanize(run.class_names[index]), float(value)) for value, index in zip(values.tolist(), indices.tolist())]


def _render_prediction_table(predictions: Sequence[Tuple[str, float]]) -> None:
    if not predictions:
        return
    frame = pd.DataFrame(predictions, columns=["class_name", "probability"])
    frame["confidence_pct"] = (frame["probability"] * 100).round(2)
    st.dataframe(
        frame[["class_name", "confidence_pct"]],
        use_container_width=True,
        hide_index=True,
    )


def _predict_image_classifier(
    image: Image.Image,
    model_name: str,
    checkpoint_path: Path,
    categories: Sequence[str],
    top_k: int,
    device: torch.device,
) -> List[Tuple[str, float]]:
    if torch is None:
        raise ModuleNotFoundError("torch")

    if model_name == "resnet50":
        model = _load_resnet_classifier(str(checkpoint_path), str(device))
    else:
        model = _load_vit_classifier(str(checkpoint_path), str(device))

    transform = _image_transform()
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs[0], dim=0)
        values, indices = torch.topk(probabilities, k=min(top_k, probabilities.numel()))

    return [
        (categories[index], float(value))
        for value, index in zip(values.tolist(), indices.tolist())
    ]


def _render_overview() -> None:
    st.subheader("Assignment 1 Workspace")
    st.write(
        "This single deployment bundles the three Assignment 1 tracks into one interface. "
        "Image and text sections expose experiment artifacts already stored in the repository, "
        "while the multimodal section uses trained checkpoints from the run directory for live inference."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Tracks", "3")
    col2.metric("Live Inference", "Multimodal")
    col3.metric("Multimodal Classes", "10")

    st.markdown("### What is included")
    st.write("- Image classification experiment dashboard")
    st.write("- Text classification experiment dashboard")
    st.write("- Multimodal results browser and predictor from saved runs")


def _render_image_tab() -> None:
    st.subheader("Image Classification")
    st.caption("Caltech-256 experiment assets and trained checkpoints packaged with the repository.")

    categories_path = IMAGE_ROOT / "artifacts" / "categories.txt"
    categories = _load_categories(str(categories_path)) if categories_path.exists() else []
    if categories_path.exists():
        class_count = len(categories)
        st.metric("Tracked categories", class_count)

    comparison_path = IMAGE_ROOT / "artifacts" / "Cmp_ResNet50_ViT.png"
    class_dist_path = IMAGE_ROOT / "artifacts" / "class_distribution.png"
    size_dist_path = IMAGE_ROOT / "artifacts" / "size_distribution.png"

    col1, col2 = st.columns(2)
    with col1:
        _display_image_if_exists(comparison_path, "ResNet50 vs ViT comparison")
        _display_image_if_exists(class_dist_path, "Class distribution")
    with col2:
        _display_image_if_exists(size_dist_path, "Image size distribution")
        screenshot_examples = _list_prediction_examples(IMAGE_ROOT / "artifacts", limit=3)
        for example in screenshot_examples:
            if example.name.startswith("Screenshot"):
                _display_image_if_exists(example, example.name)

    resnet_checkpoint = IMAGE_RUNS_ROOT / "image_resnet" / "resnet50_best.pth"
    vit_checkpoint = IMAGE_RUNS_ROOT / "image_vit" / "vit_b_16_best.pth"
    if not resnet_checkpoint.exists() and not vit_checkpoint.exists():
        st.info(
            "No deployable Caltech-256 checkpoints were found under `assignments/assignment1/run/runs_caltech256`."
        )
        return

    st.markdown("### Live Prediction")
    if torch is None or models is None or transforms is None:
        st.error("PyTorch or torchvision is not available in this environment.")
        return
    if not categories:
        st.error("Missing `categories.txt` under `image/artifacts`.")
        return

    options = []
    if resnet_checkpoint.exists():
        options.append(("ResNet50", "resnet50", resnet_checkpoint))
    if vit_checkpoint.exists():
        options.append(("ViT-B/16", "vit_b_16", vit_checkpoint))

    option_labels = [label for label, _, _ in options]
    selected_label = st.selectbox("Select image model", options=option_labels)
    selected = next(item for item in options if item[0] == selected_label)
    _, model_key, checkpoint_path = selected
    top_k = st.slider("Image top-k predictions", min_value=1, max_value=10, value=3, key="image-top-k")
    uploaded = st.file_uploader(
        "Upload an image for Caltech-256 inference",
        type=["jpg", "jpeg", "png", "webp"],
        key="image-upload",
    )

    st.caption(f"Checkpoint: `{checkpoint_path}`")
    if uploaded is None:
        return

    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    try:
        device = _resolve_device()
        predictions = _predict_image_classifier(
            image=image,
            model_name=model_key,
            checkpoint_path=checkpoint_path,
            categories=categories,
            top_k=top_k,
            device=device,
        )
        st.caption(f"Running on device: `{device}`")
        _render_prediction_table(predictions)
    except Exception as exc:
        st.exception(exc)


def _render_text_tab() -> None:
    st.subheader("Text Classification")
    st.caption("Text experiment artifacts packaged with the repository.")

    plots = [
        ("EDA plots", TEXT_ROOT / "artifacts" / "eda_plots.png"),
        ("Learning curves", TEXT_ROOT / "artifacts" / "learning_curves.png"),
        ("F1 comparison", TEXT_ROOT / "artifacts" / "f1_comparison.png"),
        ("ROC-AUC comparison", TEXT_ROOT / "artifacts" / "roc_auc_comparison.png"),
        ("Confusion matrices", TEXT_ROOT / "artifacts" / "confusion_matrices.png"),
    ]

    left, right = st.columns(2)
    for index, (caption, path) in enumerate(plots):
        with left if index % 2 == 0 else right:
            _display_image_if_exists(path, caption)

    st.info(
        "The repository includes report-ready plots for the text track, but no serialized text model "
        "checkpoint is currently packaged for cloud inference."
    )


def _render_multimodal_results(run: RunInfo, shots: int) -> None:
    summary_path = _resolve_run_asset(run, "summary.csv")
    metrics_path = _resolve_run_asset(run, "evaluation/metrics_summary.csv")
    top_failures_path = _resolve_run_asset(run, "evaluation/failed_cases/top_failures.csv")
    confusion_zero = _resolve_run_asset(run, "evaluation/confusion_matrices/zero_shot.png")
    confusion_linear = _resolve_run_asset(run, f"evaluation/confusion_matrices/linear_probe_{shots}.png")
    confusion_coop = _resolve_run_asset(run, f"evaluation/confusion_matrices/coop_{shots}.png")
    comparison_bar = _resolve_run_asset(run, "evaluation/plots/comparison_bar.png")

    col1, col2, col3 = st.columns(3)
    col1.metric("Model", run.model_dir.name)
    col2.metric("Run", run.run_dir.name)
    col3.metric("Available shots", ", ".join(str(shot) for shot in run.shots_available))

    if summary_path.exists():
        st.markdown("### Summary Metrics")
        summary_df = _load_csv(str(summary_path))
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    if metrics_path.exists():
        st.markdown("### Evaluation Table")
        metrics_df = _load_csv(str(metrics_path))
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    st.markdown("### Evaluation Visuals")
    vis1, vis2 = st.columns(2)
    with vis1:
        _display_image_if_exists(comparison_bar, "Method comparison")
        _display_image_if_exists(confusion_zero, "Zero-shot confusion matrix")
    with vis2:
        _display_image_if_exists(confusion_linear, f"Linear probe confusion matrix ({shots} shots)")
        _display_image_if_exists(confusion_coop, f"CoOp confusion matrix ({shots} shots)")

    if top_failures_path.exists():
        st.markdown("### Top Failures")
        top_failures_df = _load_csv(str(top_failures_path))
        st.dataframe(top_failures_df, use_container_width=True, hide_index=True)

    example_paths = _list_prediction_examples(_resolve_run_asset(run, "evaluation/saliency"), limit=5)
    if example_paths:
        st.markdown("### Saliency Examples")
        cols = st.columns(min(3, len(example_paths)))
        for index, path in enumerate(example_paths):
            with cols[index % len(cols)]:
                _display_image_if_exists(path, path.name, use_container_width=True)


def _render_multimodal_predictor(run: RunInfo, shots: int) -> None:
    st.markdown("### Live Prediction")

    if torch is None:
        st.error(
            "PyTorch is not available in this environment. Deploy this app with Python 3.12 "
            "and ensure the repository root `requirements.txt` is installed."
        )
        return

    method = st.radio(
        "Inference method",
        options=("linear_probe", "zero_shot"),
        horizontal=True,
    )
    top_k = st.slider("Top-k predictions", min_value=1, max_value=10, value=5)
    uploaded = st.file_uploader(
        "Upload a food image",
        type=["jpg", "jpeg", "png", "webp"],
        key=f"multimodal-upload-{run.run_dir.name}",
    )

    if uploaded is None:
        st.caption("Upload an image to run inference with the selected trained run.")
        return

    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    try:
        device = _resolve_device()
        if method == "linear_probe":
            predictions = _predict_linear_probe(image, run, shots, top_k, device)
        else:
            predictions = _predict_zero_shot(image, run, top_k, device)
        st.caption(f"Running on device: `{device}`")
        _render_prediction_table(predictions)
    except Exception as exc:
        st.exception(exc)


def _render_multimodal_tab() -> None:
    st.subheader("Multimodal")
    st.caption("Saved Food101 CLIP runs with evaluation artifacts and live inference.")

    runs_root = _default_runs_root()
    runs = _discover_runs(runs_root)
    if not runs:
        st.error(f"No valid multimodal runs found under `{runs_root}`.")
        return

    labels = list(runs.keys())
    selected_label = st.selectbox("Select trained run", options=labels, index=0)
    run = runs[selected_label]
    default_shot_index = max(0, len(run.shots_available) - 1)
    shots = st.select_slider(
        "Few-shot setting",
        options=list(run.shots_available),
        value=run.shots_available[default_shot_index],
    )

    st.write(
        f"Model ID: `{run.model_id}`  \n"
        f"Classes: {', '.join(_humanize(name) for name in run.class_names)}"
    )

    _render_multimodal_results(run, shots)
    _render_multimodal_predictor(run, shots)


def main() -> None:
    st.set_page_config(
        page_title="Learning So Deep",
        page_icon="🧠",
        layout="wide",
    )
    st.title("Learning So Deep")
    st.caption("Unified Assignment 1 deployment for image, text, and multimodal experiments.")

    overview_tab, image_tab, text_tab, multimodal_tab = st.tabs(
        ["Overview", "Image", "Text", "Multimodal"]
    )

    with overview_tab:
        _render_overview()
    with image_tab:
        _render_image_tab()
    with text_tab:
        _render_text_tab()
    with multimodal_tab:
        _render_multimodal_tab()


if __name__ == "__main__":
    main()
