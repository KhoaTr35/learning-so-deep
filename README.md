# Learning So Deep - Deep Learning Assignments

A comprehensive Python project covering **Image Classification**, **Text Classification**, and **Multimodal Learning** using modern deep learning architectures.

## Project Homepage

Visit the project homepage: [https://KhoaTr35.github.io/learning-so-deep/index.html](https://KhoaTr35.github.io/learning-so-deep/index.html)

## Project Overview

This project implements and compares various deep learning models:

### Image Classification
- **CNN** (Convolutional Neural Network)
- **Vision Transformer (ViT)**

### Text Classification
- **RNN with LSTM**
- **Transformer-based Models**

### Multimodal Classification
- **Zero-shot Learning**
- **Few-shot Learning (Prototypical Networks)**

## Quick Start

### Prerequisites
- Python 3.12+
- Virtual environment (conda/venv)

### Installation

**Manual Setup:**

1. **Clone or Setup the Repository**
   ```bash
   cd learning-so-deep
   ```

2. **Create and Activate Virtual Environment**
   ```bash
   python3.12 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Assignment Dependencies**
   ```bash
   pip install -r assignments/assignment1/requirements.txt
   ```

4. **Optional: Create Module-Specific Requirement Files**
   ```bash
   # If you add module-level requirements in the future:
   # assignments/assignment1/image/requirements.txt
   # assignments/assignment1/text/requirements.txt
   # assignments/assignment1/multimodal/requirements.txt
   ```

## Project Structure

```
.
├── README.md                              # Project overview
├── LICENSE                                # License file
├── .gitignore                             # Git ignore file
├── docs/                                  # GitHub Pages deployment assets
│   ├── index.html                         # Homepage
│   ├── assignment-1.html                  # Assignment 1 page
│   ├── assignment-2.html                  # Assignment 2 page
│   ├── assignment-3.html                  # Assignment 3 page
│   └── styles.css                         # Shared styles
├── assignments/                           # Assignment source workspace
│   ├── .streamlit/
│   │   └── config.toml                    # Streamlit configuration
│   ├── assignment1/
│   │   ├── requirements.txt               # Assignment 1 dependencies
│   │   ├── image/
│   │   │   ├── notebooks/
│   │   │   ├── reports/
│   │   │   ├── models/
│   │   │   └── artifacts/
│   │   ├── text/
│   │   │   ├── notebooks/
│   │   │   ├── reports/
│   │   │   ├── models/
│   │   │   └── artifacts/
│   │   └── multimodal/
│   │       ├── notebooks/
│   │       ├── reports/
│   │       ├── models/
│   │       └── artifacts/
│   ├── assignment2/                       # Currently empty
│   └── assignment3/                       # Currently empty
└── venv/                                  # Local virtual environment (optional)
```

### Streamlit Configuration

Settings are in `assignments/.streamlit/config.toml`:
- Theme colors
- Display settings
- Performance options


## 🛠️ Development


## Learning Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Vision Transformers Paper](https://arxiv.org/abs/2010.11929)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

## Course Information

- **Course**: Deep Learning
- **Last Updated**: March 2026

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Feel free to:
- Report issues
- Suggest improvements
- Submit pull requests
- Add new models or techniques

---