from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import streamlit as st
import torch
from PIL import Image
from torch import nn
from transformers import AutoProcessor, CLIPModel


DEFAULT_OUTPUT_ROOT = (
    Path(__file__).resolve().parent
    / "notebooks"
    / "output"
    / "clip-vit-base-patch32"
)
DEFAULT_SHOT_OPTIONS = tuple(range(8, 129, 8))


@dataclass(frozen=True)
class RunInfo:
    shots_per_class: int
    run_dir: Path
    model_id: str
    class_names: Tuple[str, ...]
    prompt_templates: Tuple[str, ...]
    checkpoint_path: Path


def _humanize(name: str) -> str:
    return name.replace("_", " ")


def _extract_tensor(outputs: object) -> torch.Tensor:
    if hasattr(outputs, "pooler_output"):
        return outputs.pooler_output
    if isinstance(outputs, (list, tuple)):
        return outputs[0]
    return outputs  # type: ignore[return-value]


def _resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _discover_runs(output_root: Path) -> Dict[int, RunInfo]:
    latest_by_shot: Dict[int, RunInfo] = {}
    if not output_root.exists():
        return latest_by_shot

    for run_dir in sorted([p for p in output_root.iterdir() if p.is_dir()]):
        config_path = run_dir / "config.json"
        checkpoint_path = run_dir / "few_shot_linear_probe.pt"
        if not config_path.exists() or not checkpoint_path.exists():
            continue

        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            shots = int(config["shots_per_class"])
            model_id = str(config["model_id"])
            class_names = tuple(config["class_names"])
            prompt_templates = tuple(config.get("prompt_templates", ["a photo of {}."]))
        except Exception:
            continue

        run = RunInfo(
            shots_per_class=shots,
            run_dir=run_dir,
            model_id=model_id,
            class_names=class_names,
            prompt_templates=prompt_templates,
            checkpoint_path=checkpoint_path,
        )
        prev = latest_by_shot.get(shots)
        if prev is None or run_dir.name > prev.run_dir.name:
            latest_by_shot[shots] = run

    return latest_by_shot


@st.cache_resource(show_spinner=False)
def _load_clip(model_id: str, device_str: str) -> Tuple[CLIPModel, AutoProcessor]:
    device = torch.device(device_str)
    processor = AutoProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).to(device)
    model.eval()
    return model, processor


@st.cache_resource(show_spinner=False)
def _load_linear_probe(checkpoint_path: str, device_str: str) -> Tuple[nn.Linear, dict]:
    device = torch.device(device_str)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    feature_dim = int(checkpoint.get("feature_dim") or checkpoint.get("in_features"))
    num_classes = int(checkpoint.get("num_classes") or len(checkpoint["class_names"]))
    probe = nn.Linear(feature_dim, num_classes)
    probe.load_state_dict(checkpoint["state_dict"])
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
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            text_outputs = model.get_text_features(**text_inputs)
            text_features = _extract_tensor(text_outputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            averaged = text_features.mean(dim=0)
            averaged = averaged / averaged.norm()
            text_feature_batches.append(averaged)

    return torch.stack(text_feature_batches, dim=0).to(device)


def _predict_few_shot(
    image: Image.Image,
    run: RunInfo,
    top_k: int,
    device: torch.device,
) -> List[Tuple[str, float]]:
    model, processor = _load_clip(run.model_id, str(device))
    probe, checkpoint = _load_linear_probe(str(run.checkpoint_path), str(device))
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

    return [(_humanize(class_names[i]), float(v)) for v, i in zip(values.tolist(), indices.tolist())]


def _predict_zero_shot(
    image: Image.Image,
    run: RunInfo,
    top_k: int,
    device: torch.device,
) -> List[Tuple[str, float]]:
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

    return [(_humanize(run.class_names[i]), float(v)) for v, i in zip(values.tolist(), indices.tolist())]


def main() -> None:
    st.set_page_config(page_title="Multimodal Classifier Demo", layout="wide")
    st.markdown("""
        <style>
        .main { background-color: #F5F5F7; }
        .stButton>button { border-radius: 20px; border: 1px solid #d1d1d6; }
        div.stProgress > div > div > div > div { background-color: #007AFF; }
        h1, h2, h3 { font-family: 'Inter', sans-serif; font-weight: 600; color: #1D1D1F; }
        </style>
    """, unsafe_allow_html=True)

    st.title("Food101 CLIP + Linear Probe Demo")

    st.caption("Upload an image and compare Few-shot Linear Probe vs Zero-shot CLIP.")

    with st.sidebar:
        st.header("Settings")
        output_root = Path(
            st.text_input("Output root", value=str(DEFAULT_OUTPUT_ROOT))
        ).expanduser()
        shot_option = st.select_slider(
            "shots_per_class",
            options=DEFAULT_SHOT_OPTIONS,
            value=128,
        )
        top_k = st.slider("Top-k predictions", min_value=1, max_value=10, value=5)

    runs_by_shot = _discover_runs(output_root)
    available = sorted(runs_by_shot.keys())
    if not available:
        st.error(f"No valid run folders found under: {output_root}")
        return

    st.info(f"Available checkpoints: {available}")
    if shot_option not in runs_by_shot:
        st.warning(
            f"No checkpoint found for shots_per_class={shot_option}. "
            f"Please choose one of: {available}"
        )
        return

    run = runs_by_shot[shot_option]
    device = _resolve_device()
    st.caption(f"Using run: `{run.run_dir.name}` | model: `{run.model_id}` | device: `{device}`")

    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])
    if uploaded is None:
        return

    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", width=360)

    with st.spinner("Running inference..."):
        few_shot_preds = _predict_few_shot(image, run, top_k=top_k, device=device)
        zero_shot_preds = _predict_zero_shot(image, run, top_k=top_k, device=device)

    # col1, col2 = st.columns(2)
    # with col1:
    #     st.subheader(f"Few-shot Linear Probe ({shot_option} shots)")
    #     for rank, (label, prob) in enumerate(few_shot_preds, start=1):
    #         st.write(f"{rank}. **{label}** - {prob:.2%}")
    # with col2:
    #     st.subheader("Zero-shot CLIP")
    #     for rank, (label, prob) in enumerate(zero_shot_preds, start=1):
    #         st.write(f"{rank}. **{label}** - {prob:.2%}")

    st.divider() # Đường kẻ ngang tinh tế
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"🎯 Few-shot ({shot_option} shots)")
        # Hiển thị kết quả bằng thanh bar
        for label, prob in few_shot_preds:
            st.write(f"{label}")
            st.progress(prob)
            st.caption(f"Confidence: {prob:.2%}")

    with col2:
        st.subheader("🌐 Zero-shot CLIP")
        for label, prob in zero_shot_preds:
            st.write(f"{label}")
            st.progress(prob)
            st.caption(f"Confidence: {prob:.2%}")

    # Thêm một phần giải thích nhỏ phía dưới
    with st.expander("🔍 Giải thích cơ chế"):
        st.write(f"**Zero-shot:** Sử dụng {len(run.prompt_templates)} mẫu câu prompt để dự đoán.")
        st.code(run.prompt_templates[0])
        st.write(f"**Few-shot:** Sử dụng một Linear Layer đã được train trên {shot_option} ảnh mỗi lớp.")


if __name__ == "__main__":
    main()
