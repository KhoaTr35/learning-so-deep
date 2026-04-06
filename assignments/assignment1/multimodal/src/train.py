from __future__ import annotations

import argparse
import copy
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoProcessor, CLIPModel

try:
    from transformers.modeling_attn_mask_utils import (
        _create_4d_causal_attention_mask,
        _prepare_4d_attention_mask,
    )
except ImportError:
    from transformers.models.clip.modeling_clip import (
        _create_4d_causal_attention_mask,
        _prepare_4d_attention_mask,
    )

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
    save_rows_csv,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run zero-shot evaluation, few-shot CoOp, and few-shot linear probe "
            "training from prepared Food101 manifests and saved CLIP embeddings."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("assignments/assignment1/multimodal/configs/food101_clip.yaml"),
    )
    return parser.parse_args()


def compute_metrics(true_labels: list[int], predicted_labels: list[int]) -> dict[str, float]:
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
    return {
        "accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "balanced_accuracy": float(balanced_accuracy_score(true_labels, predicted_labels)),
        "macro_precision": float(precision_macro),
        "macro_recall": float(recall_macro),
        "macro_f1": float(f1_macro),
        "weighted_precision": float(precision_weighted),
        "weighted_recall": float(recall_weighted),
        "weighted_f1": float(f1_weighted),
    }


def build_index(payload: dict[str, Any]) -> dict[str, int]:
    return {record_id: index for index, record_id in enumerate(payload["record_ids"])}


def select_rows(
    embedding_payload: dict[str, Any],
    index_by_record_id: dict[str, int],
    rows: list[dict[str, str]],
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    row_indices = [index_by_record_id[row["record_id"]] for row in rows]
    embeddings = embedding_payload["embeddings"][row_indices]
    label_ids = embedding_payload["label_ids"][row_indices]
    record_ids = [row["record_id"] for row in rows]
    return embeddings, label_ids, record_ids


def save_prediction_rows(
    path: Path,
    record_ids: list[str],
    true_labels: torch.Tensor,
    predicted_labels: torch.Tensor,
    scores: torch.Tensor,
    class_names: list[str],
) -> None:
    rows: list[dict[str, Any]] = []
    for index, record_id in enumerate(record_ids):
        true_label = int(true_labels[index].item())
        predicted_label = int(predicted_labels[index].item())
        rows.append(
            {
                "record_id": record_id,
                "true_label_id": true_label,
                "true_label_name": class_names[true_label],
                "predicted_label_id": predicted_label,
                "predicted_label_name": class_names[predicted_label],
                "confidence": float(scores[index].item()),
            }
        )
    save_rows_csv(
        path,
        rows,
        [
            "record_id",
            "true_label_id",
            "true_label_name",
            "predicted_label_id",
            "predicted_label_name",
            "confidence",
        ],
    )


def predict_logits(
    classifier: nn.Linear,
    embeddings: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    logits_batches: list[torch.Tensor] = []
    classifier.eval()
    with torch.inference_mode():
        for start in range(0, len(embeddings), batch_size):
            batch = embeddings[start : start + batch_size].to(device)
            logits_batches.append(classifier(batch).cpu())
    return torch.cat(logits_batches, dim=0)


def compute_similarity_logits(
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    logit_scale: float,
    batch_size: int,
) -> torch.Tensor:
    logits_batches: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(image_embeddings), batch_size):
            batch = image_embeddings[start : start + batch_size]
            logits_batches.append(logit_scale * (batch @ text_embeddings.T))
    return torch.cat(logits_batches, dim=0)


def train_linear_probe(
    train_embeddings: torch.Tensor,
    train_labels: torch.Tensor,
    val_embeddings: torch.Tensor,
    val_labels: torch.Tensor,
    num_classes: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
) -> tuple[nn.Linear, list[dict[str, Any]], dict[str, float]]:
    classifier = nn.Linear(train_embeddings.shape[1], num_classes).to(device)
    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    dataloader = DataLoader(
        TensorDataset(train_embeddings, train_labels),
        batch_size=min(batch_size, len(train_embeddings)),
        shuffle=True,
    )

    best_state = copy.deepcopy(classifier.state_dict())
    best_metrics = {"macro_f1": float("-inf")}
    history: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        classifier.train()
        running_loss = 0.0
        seen = 0
        for batch_embeddings, batch_labels in dataloader:
            batch_embeddings = batch_embeddings.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = classifier(batch_embeddings)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * batch_labels.size(0)
            seen += batch_labels.size(0)

        val_logits = predict_logits(classifier, val_embeddings, batch_size=batch_size, device=device)
        val_predictions = val_logits.argmax(dim=-1)
        val_metrics = compute_metrics(
            true_labels=val_labels.tolist(),
            predicted_labels=val_predictions.tolist(),
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": running_loss / max(seen, 1),
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
            }
        )
        if val_metrics["macro_f1"] > best_metrics["macro_f1"]:
            best_metrics = val_metrics
            best_state = {key: value.detach().cpu() for key, value in classifier.state_dict().items()}

    classifier.load_state_dict(best_state)
    return classifier, history, best_metrics


def run_zero_shot(
    test_payload: dict[str, Any],
    text_payload: dict[str, Any],
    batch_size: int,
) -> tuple[dict[str, float], torch.Tensor]:
    logit_scale = float(text_payload["logit_scale"])
    logits = compute_similarity_logits(
        image_embeddings=test_payload["embeddings"],
        text_embeddings=text_payload["class_embeddings"],
        logit_scale=logit_scale,
        batch_size=batch_size,
    )
    predictions = logits.argmax(dim=-1)
    metrics = compute_metrics(
        true_labels=test_payload["label_ids"].tolist(),
        predicted_labels=predictions.tolist(),
    )
    return metrics, logits


class CoOpPromptLearner(nn.Module):
    def __init__(
        self,
        class_names: list[str],
        tokenizer: Any,
        clip_model: CLIPModel,
        num_context_tokens: int,
        class_token_position: str,
        class_specific_context: bool,
        context_init: str | None,
        init_std: float,
    ) -> None:
        super().__init__()
        self.class_names = class_names
        self.tokenizer = tokenizer
        self.num_classes = len(class_names)
        self.num_context_tokens = num_context_tokens
        self.class_token_position = class_token_position
        self.class_specific_context = class_specific_context
        self.max_length = int(tokenizer.model_max_length)
        self.token_embedding = clip_model.text_model.embeddings.token_embedding

        context_vectors = self._initialize_context(context_init=context_init, init_std=init_std)
        if class_specific_context:
            context_vectors = context_vectors.unsqueeze(0).repeat(self.num_classes, 1, 1)
        self.context = nn.Parameter(context_vectors)

        input_ids, attention_mask, context_positions = self._build_prompt_tensors()
        self.register_buffer("input_ids", input_ids)
        self.register_buffer("attention_mask", attention_mask)
        self.register_buffer("context_positions", context_positions)

    def _initialize_context(self, context_init: str | None, init_std: float) -> torch.Tensor:
        embed_dim = int(self.token_embedding.embedding_dim)
        if context_init:
            init_ids = self.tokenizer(context_init, add_special_tokens=False)["input_ids"]
            if len(init_ids) != self.num_context_tokens:
                raise ValueError(
                    "coop.context_init must tokenize to exactly coop.num_context_tokens tokens."
                )
            init_tensor = torch.tensor(init_ids, dtype=torch.long)
            with torch.no_grad():
                return self.token_embedding(init_tensor).detach().clone()

        context = torch.empty(self.num_context_tokens, embed_dim)
        nn.init.normal_(context, std=init_std)
        return context

    def _build_prompt_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bos_id = int(self.tokenizer.bos_token_id)
        eos_id = int(self.tokenizer.eos_token_id)
        pad_id = int(self.tokenizer.pad_token_id)
        period_ids = self.tokenizer(".", add_special_tokens=False)["input_ids"]

        input_ids_rows: list[list[int]] = []
        attention_rows: list[list[int]] = []
        context_position_rows: list[list[int]] = []

        for class_name in self.class_names:
            class_ids = self.tokenizer(
                humanize_class_name(class_name),
                add_special_tokens=False,
            )["input_ids"]
            if self.class_token_position == "middle":
                ctx_before = self.num_context_tokens // 2
                ctx_after = self.num_context_tokens - ctx_before
                sequence = (
                    [bos_id]
                    + [pad_id] * ctx_before
                    + class_ids
                    + [pad_id] * ctx_after
                    + period_ids
                    + [eos_id]
                )
                context_positions = list(range(1, 1 + ctx_before))
                second_start = 1 + ctx_before + len(class_ids)
                context_positions.extend(range(second_start, second_start + ctx_after))
            else:
                sequence = (
                    [bos_id]
                    + [pad_id] * self.num_context_tokens
                    + class_ids
                    + period_ids
                    + [eos_id]
                )
                context_positions = list(range(1, 1 + self.num_context_tokens))

            if len(sequence) > self.max_length:
                raise ValueError(
                    f"Prompt for class {class_name!r} exceeds CLIP max length {self.max_length}."
                )

            attention_mask = [1] * len(sequence)
            padding_size = self.max_length - len(sequence)
            sequence = sequence + [pad_id] * padding_size
            attention_mask = attention_mask + [0] * padding_size

            input_ids_rows.append(sequence)
            attention_rows.append(attention_mask)
            context_position_rows.append(context_positions)

        return (
            torch.tensor(input_ids_rows, dtype=torch.long),
            torch.tensor(attention_rows, dtype=torch.long),
            torch.tensor(context_position_rows, dtype=torch.long),
        )

    def forward(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prompt_embeddings = self.token_embedding(self.input_ids)
        prompt_embeddings = prompt_embeddings.clone()

        if self.class_specific_context:
            context = self.context
        else:
            context = self.context.unsqueeze(0).expand(self.num_classes, -1, -1)

        batch_index = torch.arange(self.num_classes, device=prompt_embeddings.device).unsqueeze(1)
        prompt_embeddings[batch_index, self.context_positions] = context
        return self.input_ids, self.attention_mask, prompt_embeddings


def encode_text_with_prompt_embeddings(
    clip_model: CLIPModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_embeddings: torch.Tensor,
) -> torch.Tensor:
    text_model = clip_model.text_model
    hidden_states = text_model.embeddings(input_ids=input_ids, inputs_embeds=prompt_embeddings)
    causal_attention_mask = _create_4d_causal_attention_mask(
        input_ids.shape,
        hidden_states.dtype,
        device=hidden_states.device,
    )
    expanded_attention_mask = None
    if attention_mask is not None and not getattr(text_model, "_use_flash_attention_2", False):
        expanded_attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)

    encoder_outputs = text_model.encoder(
        inputs_embeds=hidden_states,
        attention_mask=expanded_attention_mask,
        causal_attention_mask=causal_attention_mask,
        output_attentions=False,
        output_hidden_states=False,
    )
    last_hidden_state = text_model.final_layer_norm(encoder_outputs.last_hidden_state)
    eos_positions = (input_ids == int(text_model.eos_token_id)).int().argmax(dim=-1)
    pooled_output = last_hidden_state[
        torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
        eos_positions,
    ]
    text_features = clip_model.text_projection(pooled_output)
    return text_features / text_features.norm(dim=-1, keepdim=True)


def build_coop_text_features(
    prompt_learner: CoOpPromptLearner,
    clip_model: CLIPModel,
    device: torch.device,
) -> torch.Tensor:
    input_ids, attention_mask, prompt_embeddings = prompt_learner()
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    prompt_embeddings = prompt_embeddings.to(device)
    return encode_text_with_prompt_embeddings(
        clip_model=clip_model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        prompt_embeddings=prompt_embeddings,
    )


def predict_coop_logits(
    prompt_learner: CoOpPromptLearner,
    clip_model: CLIPModel,
    image_embeddings: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    prompt_learner.eval()
    with torch.inference_mode():
        text_features = build_coop_text_features(prompt_learner, clip_model, device=device).cpu()
        logit_scale = float(clip_model.logit_scale.exp().detach().cpu().item())
        return compute_similarity_logits(
            image_embeddings=image_embeddings,
            text_embeddings=text_features,
            logit_scale=logit_scale,
            batch_size=batch_size,
        )


def train_coop(
    train_embeddings: torch.Tensor,
    train_labels: torch.Tensor,
    val_embeddings: torch.Tensor,
    val_labels: torch.Tensor,
    class_names: list[str],
    model_id: str,
    coop_config: dict[str, Any],
    device: torch.device,
) -> tuple[CoOpPromptLearner, CLIPModel, list[dict[str, Any]], dict[str, float]]:
    processor = AutoProcessor.from_pretrained(model_id)
    clip_model = CLIPModel.from_pretrained(model_id).to(device)
    clip_model.eval()
    for parameter in clip_model.parameters():
        parameter.requires_grad = False

    prompt_learner = CoOpPromptLearner(
        class_names=class_names,
        tokenizer=processor.tokenizer,
        clip_model=clip_model,
        num_context_tokens=int(coop_config["num_context_tokens"]),
        class_token_position=str(coop_config["class_token_position"]),
        class_specific_context=bool(coop_config["class_specific_context"]),
        context_init=coop_config.get("context_init"),
        init_std=float(coop_config["init_std"]),
    ).to(device)

    optimizer = torch.optim.AdamW(
        prompt_learner.parameters(),
        lr=float(coop_config["learning_rate"]),
        weight_decay=float(coop_config["weight_decay"]),
    )
    criterion = nn.CrossEntropyLoss()
    dataloader = DataLoader(
        TensorDataset(train_embeddings, train_labels),
        batch_size=min(int(coop_config["batch_size"]), len(train_embeddings)),
        shuffle=True,
    )

    best_state = copy.deepcopy(prompt_learner.state_dict())
    best_metrics = {"macro_f1": float("-inf")}
    history: list[dict[str, Any]] = []

    for epoch in range(1, int(coop_config["epochs"]) + 1):
        prompt_learner.train()
        running_loss = 0.0
        seen = 0
        for batch_embeddings, batch_labels in dataloader:
            batch_embeddings = batch_embeddings.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            text_features = build_coop_text_features(prompt_learner, clip_model, device=device)
            logits = clip_model.logit_scale.exp() * (batch_embeddings @ text_features.T)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * batch_labels.size(0)
            seen += batch_labels.size(0)

        val_logits = predict_coop_logits(
            prompt_learner=prompt_learner,
            clip_model=clip_model,
            image_embeddings=val_embeddings,
            batch_size=int(coop_config["batch_size"]),
            device=device,
        )
        val_predictions = val_logits.argmax(dim=-1)
        val_metrics = compute_metrics(
            true_labels=val_labels.tolist(),
            predicted_labels=val_predictions.tolist(),
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": running_loss / max(seen, 1),
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
            }
        )
        if val_metrics["macro_f1"] > best_metrics["macro_f1"]:
            best_metrics = val_metrics
            best_state = {key: value.detach().cpu() for key, value in prompt_learner.state_dict().items()}

    prompt_learner.load_state_dict(best_state)
    return prompt_learner, clip_model, history, best_metrics


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config["splits"]["seed"]))
    wandb_run = init_wandb_run(
        config=config,
        job_type="train",
        run_name=f"train-{config['model']['id'].split('/')[-1]}",
        extra_config={"config_path": str(args.config)},
    )

    model_id = config["model"]["id"]
    model_tag = sanitize_model_id(model_id)
    embedding_root = Path(config["paths"]["embedding_root"]) / model_tag
    run_root = ensure_dir(
        Path(config["paths"]["run_root"]) / model_tag / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    device = resolve_device(config["training"].get("device"))

    train_full_payload = torch.load(embedding_root / "train_full_image_embeddings.pt", map_location="cpu")
    test_payload = torch.load(embedding_root / "test_image_embeddings.pt", map_location="cpu")
    text_payload = torch.load(embedding_root / "text_embeddings.pt", map_location="cpu")
    class_names = list(text_payload["class_names"])

    zero_shot_metrics, zero_shot_logits = run_zero_shot(
        test_payload=test_payload,
        text_payload=text_payload,
        batch_size=int(config["zeroshot"]["batch_size"]),
    )
    zero_shot_probs = torch.softmax(zero_shot_logits, dim=-1)
    zero_shot_scores, zero_shot_predictions = zero_shot_probs.max(dim=-1)
    save_json(run_root / "zero_shot_metrics.json", zero_shot_metrics)
    save_prediction_rows(
        run_root / "zero_shot_predictions.csv",
        test_payload["record_ids"],
        test_payload["label_ids"],
        zero_shot_predictions,
        zero_shot_scores,
        class_names,
    )
    if wandb_run is not None:
        wandb_run.log({f"zero_shot/{key}": value for key, value in zero_shot_metrics.items()})

    train_index = build_index(train_full_payload)
    summary_rows: list[dict[str, Any]] = [{"method": "zero_shot", "shots": 0, **zero_shot_metrics}]

    for shots in [int(value) for value in config["few_shot"]["num_shots"]]:
        split_dir = Path(config["paths"]["processed_root"]) / "manifests" / f"fewshot_{shots}"
        train_rows = load_rows_csv(split_dir / "train.csv")
        val_rows = load_rows_csv(split_dir / "val.csv")
        train_embeddings, train_labels, _ = select_rows(train_full_payload, train_index, train_rows)
        val_embeddings, val_labels, _ = select_rows(train_full_payload, train_index, val_rows)

        if bool(config.get("coop", {}).get("enabled", False)):
            coop_prompt_learner, coop_clip_model, coop_history, coop_best_val_metrics = train_coop(
                train_embeddings=train_embeddings,
                train_labels=train_labels,
                val_embeddings=val_embeddings,
                val_labels=val_labels,
                class_names=class_names,
                model_id=model_id,
                coop_config=config["coop"],
                device=device,
            )
            coop_logits = predict_coop_logits(
                prompt_learner=coop_prompt_learner,
                clip_model=coop_clip_model,
                image_embeddings=test_payload["embeddings"],
                batch_size=int(config["coop"]["batch_size"]),
                device=device,
            )
            coop_predictions = coop_logits.argmax(dim=-1)
            coop_probabilities = torch.softmax(coop_logits, dim=-1)
            coop_scores, _ = coop_probabilities.max(dim=-1)
            coop_metrics = compute_metrics(
                true_labels=test_payload["label_ids"].tolist(),
                predicted_labels=coop_predictions.tolist(),
            )

            coop_dir = ensure_dir(run_root / f"fewshot_{shots}" / "coop")
            save_json(
                coop_dir / "metrics.json",
                {
                    "test": coop_metrics,
                    "best_val": coop_best_val_metrics,
                    "train_size": int(train_embeddings.shape[0]),
                    "val_size": int(val_embeddings.shape[0]),
                    "test_size": int(test_payload["embeddings"].shape[0]),
                    "class_distribution_train": dict(Counter(train_labels.tolist())),
                },
            )
            save_rows_csv(
                coop_dir / "history.csv",
                coop_history,
                ["epoch", "train_loss", "val_accuracy", "val_macro_f1"],
            )
            save_prediction_rows(
                coop_dir / "predictions.csv",
                test_payload["record_ids"],
                test_payload["label_ids"],
                coop_predictions,
                coop_scores,
                class_names,
            )
            torch.save(
                {
                    "shots": shots,
                    "class_names": class_names,
                    "coop_config": config["coop"],
                    "state_dict": {
                        key: value.detach().cpu() for key, value in coop_prompt_learner.state_dict().items()
                    },
                },
                coop_dir / "prompt_learner.pt",
            )
            summary_rows.append({"method": "coop", "shots": shots, **coop_metrics})

            if wandb_run is not None:
                for epoch_row in coop_history:
                    wandb_run.log(
                        {
                            "coop/shots": shots,
                            "coop/epoch": epoch_row["epoch"],
                            f"coop_{shots}/train_loss": epoch_row["train_loss"],
                            f"coop_{shots}/val_accuracy": epoch_row["val_accuracy"],
                            f"coop_{shots}/val_macro_f1": epoch_row["val_macro_f1"],
                        }
                    )
                wandb_run.log(
                    {
                        "coop/shots": shots,
                        **{f"coop_{shots}/test_{key}": value for key, value in coop_metrics.items()},
                        **{
                            f"coop_{shots}/best_val_{key}": value
                            for key, value in coop_best_val_metrics.items()
                        },
                    }
                )

        classifier, history, best_val_metrics = train_linear_probe(
            train_embeddings=train_embeddings,
            train_labels=train_labels,
            val_embeddings=val_embeddings,
            val_labels=val_labels,
            num_classes=len(class_names),
            batch_size=int(config["training"]["batch_size"]),
            epochs=int(config["training"]["epochs"]),
            learning_rate=float(config["training"]["learning_rate"]),
            weight_decay=float(config["training"]["weight_decay"]),
            device=device,
        )
        logits = predict_logits(
            classifier,
            test_payload["embeddings"],
            batch_size=int(config["training"]["batch_size"]),
            device=device,
        )
        predictions = logits.argmax(dim=-1)
        probs = torch.softmax(logits, dim=-1)
        scores, _ = probs.max(dim=-1)
        metrics = compute_metrics(
            true_labels=test_payload["label_ids"].tolist(),
            predicted_labels=predictions.tolist(),
        )

        probe_dir = ensure_dir(run_root / f"fewshot_{shots}" / "linear_probe")
        save_json(
            probe_dir / "metrics.json",
            {
                "test": metrics,
                "best_val": best_val_metrics,
                "train_size": int(train_embeddings.shape[0]),
                "val_size": int(val_embeddings.shape[0]),
                "test_size": int(test_payload["embeddings"].shape[0]),
                "class_distribution_train": dict(Counter(train_labels.tolist())),
            },
        )
        save_rows_csv(
            probe_dir / "history.csv",
            history,
            ["epoch", "train_loss", "val_accuracy", "val_macro_f1"],
        )
        save_prediction_rows(
            probe_dir / "predictions.csv",
            test_payload["record_ids"],
            test_payload["label_ids"],
            predictions,
            scores,
            class_names,
        )
        torch.save(
            {
                "shots": shots,
                "class_names": class_names,
                "state_dict": {key: value.detach().cpu() for key, value in classifier.state_dict().items()},
            },
            probe_dir / "linear_probe.pt",
        )
        summary_rows.append({"method": "few_shot_linear_probe", "shots": shots, **metrics})

        if wandb_run is not None:
            for epoch_row in history:
                wandb_run.log(
                    {
                        "few_shot_linear_probe/shots": shots,
                        "few_shot_linear_probe/epoch": epoch_row["epoch"],
                        f"few_shot_linear_probe_{shots}/train_loss": epoch_row["train_loss"],
                        f"few_shot_linear_probe_{shots}/val_accuracy": epoch_row["val_accuracy"],
                        f"few_shot_linear_probe_{shots}/val_macro_f1": epoch_row["val_macro_f1"],
                    }
                )
            wandb_run.log(
                {
                    "few_shot_linear_probe/shots": shots,
                    **{
                        f"few_shot_linear_probe_{shots}/test_{key}": value
                        for key, value in metrics.items()
                    },
                    **{
                        f"few_shot_linear_probe_{shots}/best_val_{key}": value
                        for key, value in best_val_metrics.items()
                    },
                }
            )

    save_rows_csv(
        run_root / "summary.csv",
        summary_rows,
        [
            "method",
            "shots",
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
    save_json(
        run_root / "run_config.json",
        {
            "config_path": str(args.config),
            "device": str(device),
            "model_id": model_id,
        },
    )

    if wandb_run is not None:
        log_wandb_artifact(
            wandb_run,
            artifact_name=f"{model_tag}_training_run",
            artifact_type="training-results",
            path=run_root,
        )
        finish_wandb_run(
            wandb_run,
            {
                "run_root": str(run_root),
                "model_id": model_id,
                "zero_shot": zero_shot_metrics,
            },
        )

    print(f"Saved training artifacts to {run_root}")


if __name__ == "__main__":
    main()
