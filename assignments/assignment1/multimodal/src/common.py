from __future__ import annotations

import csv
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from dotenv import load_dotenv
import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping in {path}, received {type(config)!r}")

    prompts = config.get("prompts", {})
    templates = list(prompts.get("templates", []))
    num_sample_prompt = int(prompts.get("num_sample_prompt", len(templates)))
    if num_sample_prompt <= 0:
        raise ValueError("prompts.num_sample_prompt must be positive.")
    if len(templates) < num_sample_prompt:
        raise ValueError(
            "prompts.templates must contain at least prompts.num_sample_prompt entries."
        )
    if any("{}" not in template for template in templates[:num_sample_prompt]):
        raise ValueError("Every prompt template must include a {} placeholder.")

    classes = [canonicalize_class_name(name) for name in config["dataset"]["selected_classes"]]
    if len(classes) != len(set(classes)):
        raise ValueError("dataset.selected_classes must be unique.")
    if len(classes) != 10:
        raise ValueError("dataset.selected_classes must contain exactly 10 classes.")

    num_shots = [int(value) for value in config.get("few_shot", {}).get("num_shots", [])]
    if not num_shots:
        raise ValueError("few_shot.num_shots must include at least one value.")
    if any(value <= 0 for value in num_shots):
        raise ValueError("few_shot.num_shots values must be positive.")

    wandb_config = config.get("wandb", {})
    if wandb_config.get("enabled", False):
        if not wandb_config.get("project"):
            raise ValueError("wandb.project is required when wandb.enabled is true.")

    coop_config = config.get("coop", {})
    if coop_config.get("enabled", False):
        if int(coop_config.get("num_context_tokens", 0)) <= 0:
            raise ValueError("coop.num_context_tokens must be positive.")
        if coop_config.get("class_token_position") not in {"middle", "end"}:
            raise ValueError("coop.class_token_position must be 'middle' or 'end'.")
        if int(coop_config.get("batch_size", 0)) <= 0:
            raise ValueError("coop.batch_size must be positive.")
        if int(coop_config.get("epochs", 0)) <= 0:
            raise ValueError("coop.epochs must be positive.")
        if float(coop_config.get("learning_rate", 0.0)) <= 0:
            raise ValueError("coop.learning_rate must be positive.")
        if float(coop_config.get("weight_decay", -1.0)) < 0:
            raise ValueError("coop.weight_decay must be zero or positive.")
        if float(coop_config.get("init_std", 0.0)) <= 0:
            raise ValueError("coop.init_std must be positive.")

    return config


def canonicalize_class_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def humanize_class_name(name: str) -> str:
    return name.replace("_", " ")


def sanitize_model_id(model_id: str) -> str:
    return model_id.replace("/", "_")


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def load_env(env_path: str | Path | None = None) -> None:
    if env_path:
        load_dotenv(dotenv_path=Path(env_path), override=False)
        return

    candidates = [
        Path.cwd() / ".env",
        Path("assignments/assignment1/multimodal/.env"),
    ]
    for candidate in candidates:
        if candidate.exists():
            load_dotenv(dotenv_path=candidate, override=False)
            break


def flatten_config(prefix: str, value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        flattened: dict[str, Any] = {}
        for key, inner_value in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(flatten_config(next_prefix, inner_value))
        return flattened
    return {prefix: value}


def init_wandb_run(
    config: dict[str, Any],
    job_type: str,
    run_name: str,
    extra_config: dict[str, Any] | None = None,
):
    wandb_config = config.get("wandb", {})
    if not wandb_config.get("enabled", False):
        return None

    load_env()
    if not os.environ.get("WANDB_API_KEY") and wandb_config.get("mode", "online") == "online":
        raise ValueError(
            "WANDB_API_KEY is not set. Put it in assignments/assignment1/multimodal/.env "
            "or export it in your shell."
        )

    import wandb

    merged_config = flatten_config("", config)
    if extra_config:
        merged_config.update(flatten_config("", extra_config))

    return wandb.init(
        project=wandb_config["project"],
        entity=wandb_config.get("entity"),
        job_type=job_type,
        name=run_name,
        mode=wandb_config.get("mode", "online"),
        tags=list(wandb_config.get("tags", [])) + [job_type],
        config=merged_config,
    )


def log_wandb_artifact(run: Any, artifact_name: str, artifact_type: str, path: str | Path) -> None:
    if run is None:
        return

    import wandb

    target = Path(path)
    artifact = wandb.Artifact(artifact_name, type=artifact_type)
    if target.is_dir():
        artifact.add_dir(str(target))
    else:
        artifact.add_file(str(target))
    run.log_artifact(artifact)


def finish_wandb_run(run: Any, summary: dict[str, Any] | None = None) -> None:
    if run is None:
        return
    if summary:
        for key, value in flatten_config("", summary).items():
            run.summary[key] = value
    run.finish()


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def save_rows_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_rows_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested_device: str | None = None) -> torch.device:
    if requested_device:
        return torch.device(requested_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_few_shot_indices(
    labels: list[int],
    selected_original_ids: list[int],
    shots_per_class: int,
    val_per_class: int,
    seed: int,
) -> tuple[list[int], list[int]]:
    grouped_indices: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        grouped_indices[int(label)].append(index)

    rng = random.Random(seed + shots_per_class)
    train_indices: list[int] = []
    val_indices: list[int] = []
    required_examples = shots_per_class + val_per_class

    for original_label in selected_original_ids:
        indices = list(grouped_indices[original_label])
        rng.shuffle(indices)
        if len(indices) < required_examples:
            raise ValueError(
                f"Class {original_label} has {len(indices)} samples, required {required_examples}."
            )
        train_indices.extend(indices[:shots_per_class])
        val_indices.extend(indices[shots_per_class:required_examples])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def resolve_run_dir(run_root: str | Path, run_dir: str | Path | None = None) -> Path:
    if run_dir is not None:
        resolved = Path(run_dir)
        if not resolved.exists():
            raise FileNotFoundError(f"Run directory does not exist: {resolved}")
        return resolved

    root = Path(run_root)
    if not root.exists():
        raise FileNotFoundError(f"Run root does not exist: {root}")

    candidates = sorted([path for path in root.iterdir() if path.is_dir()])
    if not candidates:
        raise FileNotFoundError(f"No run directories found under: {root}")
    return candidates[-1]
