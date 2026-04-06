from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import nn
from transformers import AutoProcessor, CLIPModel

from common import (
    load_config,
    resolve_device,
    resolve_run_dir,
    sanitize_model_id,
)
from train import CoOpPromptLearner, build_coop_text_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run inference with Food101 zero-shot, linear probe, or CoOp checkpoints."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("assignments/assignment1/multimodal/configs/food101_clip.yaml"),
    )
    parser.add_argument("--image", type=Path, required=True, help="Path to an input image.")
    parser.add_argument(
        "--method",
        choices=("zeroshot", "linear_probe", "coop"),
        required=True,
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=None,
        help="Required for few-shot methods.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Explicit training run directory. Defaults to the latest run.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def extract_single_image_embedding(
    model: CLIPModel,
    processor: AutoProcessor,
    image_path: Path,
    device: torch.device,
) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    with torch.inference_mode():
        features = model.get_image_features(pixel_values=pixel_values)
        if hasattr(features, "pooler_output"):
            features = features.pooler_output
        elif isinstance(features, (list, tuple)):
            features = features[0]
        features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu()


def load_linear_probe(
    checkpoint_path: Path,
    embedding_dim: int,
    num_classes: int,
) -> nn.Linear:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    classifier = nn.Linear(embedding_dim, num_classes)
    classifier.load_state_dict(checkpoint["state_dict"])
    classifier.eval()
    return classifier


def load_coop_prompt_learner(
    checkpoint_path: Path,
    clip_model: CLIPModel,
    processor: AutoProcessor,
    class_names: list[str],
) -> CoOpPromptLearner:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    coop_config = checkpoint["coop_config"]
    prompt_learner = CoOpPromptLearner(
        class_names=class_names,
        tokenizer=processor.tokenizer,
        clip_model=clip_model,
        num_context_tokens=int(coop_config["num_context_tokens"]),
        class_token_position=str(coop_config["class_token_position"]),
        class_specific_context=bool(coop_config["class_specific_context"]),
        context_init=coop_config.get("context_init"),
        init_std=float(coop_config["init_std"]),
    )
    prompt_learner.load_state_dict(checkpoint["state_dict"])
    prompt_learner.eval()
    return prompt_learner


def top_predictions(
    logits: torch.Tensor,
    class_names: list[str],
    top_k: int,
) -> list[dict[str, Any]]:
    probabilities = torch.softmax(logits, dim=-1)[0]
    scores, indices = torch.topk(probabilities, k=min(top_k, len(class_names)))
    return [
        {
            "rank": rank + 1,
            "class_id": int(class_index),
            "class_name": class_names[int(class_index)],
            "probability": float(score),
        }
        for rank, (score, class_index) in enumerate(zip(scores.tolist(), indices.tolist()))
    ]


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = resolve_device(config["training"].get("device"))
    model_id = config["model"]["id"]
    model_tag = sanitize_model_id(model_id)
    run_root = Path(config["paths"]["run_root"]) / model_tag
    run_dir = resolve_run_dir(run_root=run_root, run_dir=args.run_dir)
    embedding_root = Path(config["paths"]["embedding_root"]) / model_tag

    processor = AutoProcessor.from_pretrained(model_id)
    clip_model = CLIPModel.from_pretrained(model_id).to(device)
    clip_model.eval()
    class_names = list(config["dataset"]["selected_classes"])

    image_embedding = extract_single_image_embedding(
        model=clip_model,
        processor=processor,
        image_path=args.image,
        device=device,
    )

    if args.method == "zeroshot":
        text_payload = torch.load(embedding_root / "text_embeddings.pt", map_location="cpu")
        logits = float(text_payload["logit_scale"]) * (image_embedding @ text_payload["class_embeddings"].T)
    elif args.method == "linear_probe":
        if args.shots is None:
            raise ValueError("--shots is required for method=linear_probe")
        checkpoint_path = run_dir / f"fewshot_{args.shots}" / "linear_probe" / "linear_probe.pt"
        classifier = load_linear_probe(
            checkpoint_path=checkpoint_path,
            embedding_dim=int(image_embedding.shape[1]),
            num_classes=len(class_names),
        )
        with torch.inference_mode():
            logits = classifier(image_embedding)
    else:
        if args.shots is None:
            raise ValueError("--shots is required for method=coop")
        checkpoint_path = run_dir / f"fewshot_{args.shots}" / "coop" / "prompt_learner.pt"
        prompt_learner = load_coop_prompt_learner(
            checkpoint_path=checkpoint_path,
            clip_model=clip_model,
            processor=processor,
            class_names=class_names,
        ).to(device)
        with torch.inference_mode():
            text_features = build_coop_text_features(prompt_learner, clip_model, device=device).cpu()
            logits = clip_model.logit_scale.exp().detach().cpu() * (image_embedding @ text_features.T)

    result = {
        "image": str(args.image),
        "run_dir": str(run_dir),
        "method": args.method,
        "shots": args.shots,
        "predictions": top_predictions(logits=logits, class_names=class_names, top_k=args.top_k),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
