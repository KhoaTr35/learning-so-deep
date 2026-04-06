# Learning So Deep

Deep learning coursework repository with:
- source workspaces under `assignments/`
- published report pages under `docs/`
- active implementation for Assignment 1 across image, text, and multimodal tracks

Project site: [https://KhoaTr35.github.io/learning-so-deep/](https://KhoaTr35.github.io/learning-so-deep/)

## Overview

This repository is organized in two layers:

- `assignments/` contains the working code, notebooks, reports, models, and generated artifacts.
- `docs/` contains the static GitHub Pages site that presents the assignment reports.

Current status:
- `assignment1` is the only assignment with implementation assets in the repository.
- `assignment2` and `assignment3` currently exist as workspace placeholders in `assignments/`, while their report-design pages already exist in `docs/`.

## Assignment Coverage

### Assignment 1

Assignment 1 is split into three tracks:

- `image`: image classification work, notebooks, Streamlit app, reports, models, and artifacts
- `text`: text classification reports, models, notebooks, and artifacts
- `multimodal`: CLIP-based Food101 pipeline with preprocessing, embedding extraction, training, evaluation, inference, notebooks, Streamlit app, and generated artifacts

### Assignment 2

- `assignments/assignment2/` currently has no tracked implementation files
- `docs/assignment-2/` already contains the report-design page for a robustness and generalization study

### Assignment 3

- `assignments/assignment3/` currently has no tracked implementation files
- `docs/assignment-3/` already contains the report-design page for deployment and model optimization

## Quick Start

### Prerequisites

- Python 3.12+
- `venv` or another virtual environment tool

### Install

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r assignments/assignment1/requirements.txt
```

The shared dependency file currently used by Assignment 1 includes:
- `streamlit`
- `transformers`
- `torch`
- `datasets`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `numpy`
- `pandas`
- `PyYAML`
- `python-dotenv`
- `wandb`

## Common Entry Points

### Static site

Open the published site or inspect local files in `docs/`:

- `docs/index.html`
- `docs/assignment-1/`
- `docs/assignment-2/`
- `docs/assignment-3/`

### Assignment 1 multimodal pipeline

From the repository root:

```bash
python3 assignments/assignment1/multimodal/src/download.py \
  --output-dir assignments/assignment1/multimodal/artifacts/data/food101
python3 assignments/assignment1/multimodal/src/preprocess.py
python3 assignments/assignment1/multimodal/src/extract_embedding.py
python3 assignments/assignment1/multimodal/src/train.py
python3 assignments/assignment1/multimodal/src/evaluate.py
```

Optional WandB setup:

```bash
cp assignments/assignment1/multimodal/.env.example assignments/assignment1/multimodal/.env
```

Then set `WANDB_API_KEY` in that `.env` file, or disable WandB in `assignments/assignment1/multimodal/configs/food101_clip.yaml`.

### Streamlit apps

The repository currently includes these Streamlit entry points:

- `assignments/assignment1/image/streamlit_app.py`
- `assignments/assignment1/multimodal/streamlit_app.py`

Shared Streamlit configuration lives in `assignments/.streamlit/config.toml`.

## Repository Structure

```text
.
├── README.md
├── LICENSE
├── .gitignore
├── assignments/
│   ├── .streamlit/
│   │   └── config.toml
│   ├── assignment1/
│   │   ├── requirements.txt
│   │   ├── image/
│   │   │   ├── artifacts/
│   │   │   ├── models/
│   │   │   ├── notebooks/
│   │   │   ├── reports/
│   │   │   └── streamlit_app.py
│   │   ├── text/
│   │   │   ├── artifacts/
│   │   │   ├── models/
│   │   │   ├── notebooks/
│   │   │   └── reports/
│   │   └── multimodal/
│   │       ├── artifacts/
│   │       ├── configs/
│   │       ├── models/
│   │       ├── notebooks/
│   │       ├── reports/
│   │       ├── src/
│   │       ├── README.md
│   │       ├── build_report_data.py
│   │       ├── food101_eda.py
│   │       └── streamlit_app.py
│   ├── assignment2/
│   └── assignment3/
├── docs/
│   ├── index.html
│   ├── styles.css
│   ├── assignment-1/
│   │   ├── index.html
│   │   ├── image/
│   │   ├── multimodal/
│   │   └── text/
│   ├── assignment-2/
│   │   └── index.html
│   └── assignment-3/
│       └── index.html
├── venv/
└── wandb/
```

## Notes

- `venv/` and `wandb/` are local runtime directories present in this workspace.
- Some generated files such as `.DS_Store`, `__pycache__`, and environment files exist inside assignment folders but are omitted from the structure above for clarity.
- The root README describes the repository at a high level. Track-specific details for the multimodal workflow are documented in `assignments/assignment1/multimodal/README.md`.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
