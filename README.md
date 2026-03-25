# Learning So Deep - Deep Learning Assignments

A comprehensive Python project covering **Image Classification**, **Text Classification**, and **Multimodal Learning** using modern deep learning architectures.

## Project Homepage

Visit the project homepage: [https://KhoaTr35.github.io/learning-so-deep/index.html](Home Page)

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
- PyTorch 2.0+
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

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Module-Specific Dependencies (Optional)**
   ```bash
   pip install -r docs/code/assignment1/image_classification/requirements.txt
   pip install -r docs/code/assignment1/text_classification/requirements.txt
   pip install -r docs/code/assignment1/multimodal_classification/requirements.txt
   ```

## Project Structure

```
.
├── README.md                              # Project overview
├── LICENSE                                # License file
├── docs/                                  # GitHub Pages deployment folder
│   ├── index.html                         # Homepage
│   ├── assignment-1.html                  # Assignment 1 page
│   ├── assignment-2.html                  # Assignment 2 page
│   ├── assignment-3.html                  # Assignment 3 page
│   ├── styles.css                         # Styling
│   └── code/                              # Source code
│   │   ├── assignment1/
│   │   │   ├── image_classification/
│   │   │   │   ├── requirements.txt
│   │   │   ├── text_classification/
│   │   │   │   ├── requirements.txt
│   │   │   └── multimodal_classification/
│   │   │       ├── requirements.txt
│   │   ├── assignment2/
│   │   └── assignment3/
├── pages/                                 # Markdown documentation
│   ├── index.md                           # Landing page
│   ├── assignment-1.md                    # Assignment 1 documentation
│   ├── assignment-2.md                    # Assignment 2 documentation
│   ├── assignment-3.md                    # Assignment 3 documentation
│   └── code/                              # Assignment code directory
├── requirements.txt                       # Root dependencies
└── .gitignore                             # Git ignore file
```

### Streamlit Configuration

Settings are in `github-pages/.streamlit/config.toml`:
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