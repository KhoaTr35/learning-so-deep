# Multimodal Food101 Pipeline

This folder contains the full CLIP-based multimodal pipeline for `ethz/food101` on 10 selected classes. It supports:
- full dataset download
- preprocessing and few-shot split generation
- image and text embedding extraction
- zero-shot evaluation
- few-shot linear probe training
- few-shot CoOp training
- WandB logging
- inference on new images
- final evaluation reports with plots, confusion matrices, CoOp token decoding, and saliency maps

## 1. Install

From the repo root:

```bash
pip install -r assignments/assignment1/requirements.txt
```

Optional WandB setup:

```bash
cp assignments/assignment1/multimodal/.env.example assignments/assignment1/multimodal/.env
```

Then set:

```bash
WANDB_API_KEY=your_api_key
```

If you do not want WandB, set `wandb.enabled: false` or `wandb.mode: disabled` in [configs/food101_clip.yaml](/Users/khoatran/Developer/learning-so-deep/assignments/assignment1/multimodal/configs/food101_clip.yaml).

## 2. Main Config

The main config is [configs/food101_clip.yaml](/Users/khoatran/Developer/learning-so-deep/assignments/assignment1/multimodal/configs/food101_clip.yaml).

It controls:
- `model.id`
- `dataset.name`
- `dataset.selected_classes`
- `prompts.num_sample_prompt` and `prompts.templates`
- `few_shot.num_shots`
- split seed and validation size
- zero-shot batch size
- linear probe training config
- CoOp config:
  `num_context_tokens`, `class_token_position`, `class_specific_context`, `context_init`, optimizer settings
- output paths
- WandB project/entity/mode/tags

## 3. End-to-End Run

Run from the repo root in this order.

### 3.1 Download dataset

```bash
python3 assignments/assignment1/multimodal/src/download.py \
  --output-dir assignments/assignment1/multimodal/artifacts/data/food101
```

This exports Food101 images and labels to disk.

### 3.2 Preprocess and create few-shot splits

```bash
python3 assignments/assignment1/multimodal/src/preprocess.py
```

Outputs:
- filtered images for the 10 selected classes
- `train_full.csv`
- `test.csv`
- `fewshot_{8,16,32,64,128}/{train,val}.csv`

Default output root:
[artifacts/processed_food101_clip](/Users/khoatran/Developer/learning-so-deep/assignments/assignment1/multimodal/artifacts/processed_food101_clip)

### 3.3 Extract CLIP embeddings

```bash
python3 assignments/assignment1/multimodal/src/extract_embedding.py
```

Outputs:
- `train_full_image_embeddings.pt`
- `test_image_embeddings.pt`
- `text_embeddings.pt`
- `prompt_metadata.json`

Zero-shot uses the average of 5 prompt embeddings to represent each class.

Default output root:
[artifacts/embeddings_food101_clip](/Users/khoatran/Developer/learning-so-deep/assignments/assignment1/multimodal/artifacts/embeddings_food101_clip)

### 3.4 Train and evaluate

```bash
python3 assignments/assignment1/multimodal/src/train.py
```

This runs:
- zero-shot
- CoOp for each few-shot setting if `coop.enabled: true`
- linear probe for each few-shot setting

Run outputs:
- zero-shot metrics and predictions
- `fewshot_<k>/coop/`
- `fewshot_<k>/linear_probe/`
- `summary.csv`
- `run_config.json`

Default run root:
[artifacts/runs_food101_clip](/Users/khoatran/Developer/learning-so-deep/assignments/assignment1/multimodal/artifacts/runs_food101_clip)

## 4. Inference

Use [src/infer.py](/Users/khoatran/Developer/learning-so-deep/assignments/assignment1/multimodal/src/infer.py) for a single image.

Zero-shot:

```bash
python3 assignments/assignment1/multimodal/src/infer.py \
  --image path/to/image.jpg \
  --method zeroshot
```

Linear probe:

```bash
python3 assignments/assignment1/multimodal/src/infer.py \
  --image path/to/image.jpg \
  --method linear_probe \
  --shots 32
```

CoOp:

```bash
python3 assignments/assignment1/multimodal/src/infer.py \
  --image path/to/image.jpg \
  --method coop \
  --shots 32
```

Optional:
- `--run-dir <path>` to choose a specific run
- `--top-k 5`

## 5. Final Evaluation Report

Generate plots and analysis for a saved run:

```bash
python3 assignments/assignment1/multimodal/src/evaluate.py
```

Or for a specific run:

```bash
python3 assignments/assignment1/multimodal/src/evaluate.py \
  --run-dir assignments/assignment1/multimodal/artifacts/runs_food101_clip/openai_clip-vit-base-patch32/<timestamp>
```

Outputs under `evaluation/`:
- `metrics_summary.csv`
- comparison bar plot
- confusion matrices for zero-shot, CoOp, and linear probe
- decoded CoOp context vectors
- `top_failures.csv`
- saliency maps for top failed predictions

## 6. Artifact Layout

Processed data:
- [artifacts/processed_food101_clip](/Users/khoatran/Developer/learning-so-deep/assignments/assignment1/multimodal/artifacts/processed_food101_clip)

Embeddings:
- [artifacts/embeddings_food101_clip](/Users/khoatran/Developer/learning-so-deep/assignments/assignment1/multimodal/artifacts/embeddings_food101_clip)

Training runs:
- [artifacts/runs_food101_clip](/Users/khoatran/Developer/learning-so-deep/assignments/assignment1/multimodal/artifacts/runs_food101_clip)

## 7. Minimal Command Sequence

```bash
pip install -r assignments/assignment1/requirements.txt
python3 assignments/assignment1/multimodal/src/download.py --output-dir assignments/assignment1/multimodal/artifacts/data/food101
python3 assignments/assignment1/multimodal/src/preprocess.py
python3 assignments/assignment1/multimodal/src/extract_embedding.py
python3 assignments/assignment1/multimodal/src/train.py
python3 assignments/assignment1/multimodal/src/evaluate.py
```

## 8. Notes

- `train.py`, `infer.py`, and `evaluate.py` require the CLIP model to be available locally or downloadable from Hugging Face.
- WandB upload requires network access and `WANDB_API_KEY`.
- The latest run is selected automatically if `--run-dir` is not provided.
- CoOp is trained with frozen CLIP and learnable context tokens only.
