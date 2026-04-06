from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, CLIPModel

from common import (
    ensure_dir,
    finish_wandb_run,
    humanize_class_name,
    init_wandb_run,
    load_config,
    load_rows_csv,
    log_wandb_artifact,
    resolve_device,
    sanitize_model_id,
    save_json,
    set_seed,
)


class ManifestImageDataset(Dataset):
    def __init__(self, processed_root: Path, rows: list[dict[str, str]]) -> None:
        self.processed_root = processed_root
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        image_path = self.processed_root / row["image_path"]
        image = Image.open(image_path).convert("RGB")
        return {
            "record_id": row["record_id"],
            "image": image,
            "label_id": int(row["selected_label_id"]),
            "label_name": row["selected_label_name"],
            "image_path": row["image_path"],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract CLIP embeddings for prepared Food101 images and prompts."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("assignments/assignment1/multimodal/configs/food101_clip.yaml"),
    )
    return parser.parse_args()


def collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "record_ids": [item["record_id"] for item in batch],
        "images": [item["image"] for item in batch],
        "label_ids": torch.tensor([item["label_id"] for item in batch], dtype=torch.long),
        "label_names": [item["label_name"] for item in batch],
        "image_paths": [item["image_path"] for item in batch],
    }


def extract_image_embeddings(
    model: CLIPModel,
    processor: AutoProcessor,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    embedding_batches: list[torch.Tensor] = []
    label_batches: list[torch.Tensor] = []
    record_ids: list[str] = []
    label_names: list[str] = []
    image_paths: list[str] = []

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

            features = image_features / image_features.norm(dim=-1, keepdim=True)
            embedding_batches.append(features.cpu())
            label_batches.append(batch["label_ids"].cpu())
            record_ids.extend(batch["record_ids"])
            label_names.extend(batch["label_names"])
            image_paths.extend(batch["image_paths"])

    return {
        "embeddings": torch.cat(embedding_batches, dim=0),
        "label_ids": torch.cat(label_batches, dim=0),
        "record_ids": record_ids,
        "label_names": label_names,
        "image_paths": image_paths,
    }


def build_text_embeddings(
    model: CLIPModel,
    processor: AutoProcessor,
    class_names: list[str],
    prompt_templates: list[str],
    device: torch.device,
) -> dict[str, Any]:
    prompt_rows: list[dict[str, Any]] = []
    class_embeddings: list[torch.Tensor] = []

    model.eval()
    with torch.inference_mode():
        for label_id, class_name in enumerate(class_names):
            prompts = [template.format(humanize_class_name(class_name)) for template in prompt_templates]
            text_inputs = processor(
                text=prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            text_inputs = {key: value.to(device) for key, value in text_inputs.items()}

            # Lấy text features
            text_outputs = model.get_text_features(**text_inputs)

            # Kiểm tra và trích xuất Tensor từ Object
            if hasattr(text_outputs, "pooler_output"):
                text_features = text_outputs.pooler_output
            elif isinstance(text_outputs, (list, tuple)):
                text_features = text_outputs[0]
            else:
                text_features = text_outputs
                
            prompt_embeddings = text_features / text_features.norm(dim=-1, keepdim=True)

            # average 5 prompt embeddings to get class embedding
            pooled_embedding = prompt_embeddings.mean(dim=0)

            # norm again 
            pooled_embedding = pooled_embedding / pooled_embedding.norm()
            class_embeddings.append(pooled_embedding.cpu())

            for prompt_index, prompt in enumerate(prompts):
                prompt_rows.append(
                    {
                        "label_id": label_id,
                        "label_name": class_name,
                        "prompt_index": prompt_index,
                        "prompt_text": prompt,
                        "embedding": prompt_embeddings[prompt_index].cpu(),
                    }
                )

    return {
        "class_names": class_names,
        "prompt_templates": prompt_templates,
        "class_embeddings": torch.stack(class_embeddings, dim=0),
        "prompt_rows": prompt_rows,
        "logit_scale": float(model.logit_scale.exp().detach().cpu().item()),
    }


def save_prompt_metadata(output_dir: Path, prompt_rows: list[dict[str, Any]]) -> None:
    serializable_rows = [
        {
            "label_id": row["label_id"],
            "label_name": row["label_name"],
            "prompt_index": row["prompt_index"],
            "prompt_text": row["prompt_text"],
        }
        for row in prompt_rows
    ]
    save_json(output_dir / "prompt_metadata.json", {"rows": serializable_rows})


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config["splits"]["seed"]))
    wandb_run = init_wandb_run(
        config=config,
        job_type="extract-embeddings",
        run_name=f"extract-{config['model']['id'].split('/')[-1]}",
        extra_config={"config_path": str(args.config)},
    )

    processed_root = Path(config["paths"]["processed_root"])
    manifests_root = processed_root / "manifests"
    model_id = config["model"]["id"]
    model_tag = sanitize_model_id(model_id)
    output_root = ensure_dir(Path(config["paths"]["embedding_root"]) / model_tag)
    device = resolve_device(config["training"].get("device"))

    processor = AutoProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).to(device)
    prompt_templates = list(config["prompts"]["templates"])[: int(config["prompts"]["num_sample_prompt"])]
    class_names = list(config["dataset"]["selected_classes"])

    for split_name in ("train_full", "test"):
        rows = load_rows_csv(manifests_root / f"{split_name}.csv")
        dataset = ManifestImageDataset(processed_root=processed_root, rows=rows)
        dataloader = DataLoader(
            dataset,
            batch_size=int(config["zeroshot"]["batch_size"]),
            shuffle=False,
            num_workers=int(config["training"]["num_workers"]),
            collate_fn=collate_batch,
            pin_memory=torch.cuda.is_available(),
        )
        payload = extract_image_embeddings(
            model=model,
            processor=processor,
            dataloader=dataloader,
            device=device,
        )
        torch.save(payload, output_root / f"{split_name}_image_embeddings.pt")
        if wandb_run is not None:
            wandb_run.log(
                {
                    f"embeddings/{split_name}_num_samples": len(payload["record_ids"]),
                    f"embeddings/{split_name}_embedding_dim": int(payload["embeddings"].shape[1]),
                }
            )

    text_payload = build_text_embeddings(
        model=model,
        processor=processor,
        class_names=class_names,
        prompt_templates=prompt_templates,
        device=device,
    )
    prompt_embedding_tensor = torch.stack(
        [row["embedding"] for row in text_payload["prompt_rows"]],
        dim=0,
    )
    torch.save(
        {
            "class_names": text_payload["class_names"],
            "prompt_templates": text_payload["prompt_templates"],
            "class_embeddings": text_payload["class_embeddings"],
            "prompt_embeddings": prompt_embedding_tensor,
            "logit_scale": text_payload["logit_scale"],
        },
        output_root / "text_embeddings.pt",
    )
    save_prompt_metadata(output_root, text_payload["prompt_rows"])
    if wandb_run is not None:
        wandb_run.log(
            {
                "embeddings/num_classes": len(class_names),
                "embeddings/num_prompts": len(text_payload["prompt_rows"]),
                "embeddings/text_embedding_dim": int(text_payload["class_embeddings"].shape[1]),
            }
        )
        log_wandb_artifact(
            wandb_run,
            artifact_name=f"{model_tag}_embeddings",
            artifact_type="embeddings",
            path=output_root,
        )
        finish_wandb_run(
            wandb_run,
            {
                "embedding_root": str(output_root),
                "model_id": model_id,
            },
        )

    print(f"Saved embeddings to {output_root}")


if __name__ == "__main__":
    main()
