# Learning So Deep - Deep Learning Assignments

A comprehensive Python project covering **Image Classification**, **Text Classification**, and **Multimodal Learning** using modern deep learning architectures.

## 📚 Project Overview

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

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- PyTorch 2.0+
- Virtual environment (conda/venv)

### Installation

1. **Clone or Setup the Repository**
   ```bash
   cd learning-so-deep
   ```

2. **Create and Activate Virtual Environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Module-Specific Dependencies (Optional)**
   ```bash
   pip install -r github-pages/code/image_classification/requirements.txt
   pip install -r github-pages/code/text_classification/requirements.txt
   pip install -r github-pages/code/multimodal_classification/requirements.txt
   ```

## 📁 Project Structure

```
.
├── README.md                              # Project overview
├── requirements.txt                       # Root dependencies
├── .gitignore                             # Git ignore file
├── LICENSE                                # MIT License
├── venv/                                  # Python virtual environment
├── github-pages/                          # Main project directory
│   ├── index.md                           # Landing page
│   ├── assignment-1.md                    # Assignment documentation
│   ├── requirements.txt                   # Common dependencies
│   ├── code/                              # Source code
│   │   ├── image_classification/
│   │   │   ├── data_preprocessing.py     # Image preprocessing
│   │   │   ├── cnn_model.py              # CNN implementation
│   │   │   ├── vit_model.py              # Vision Transformer
│   │   │   ├── train.py                  # Training script
│   │   │   ├── evaluate.py               # Evaluation script
│   │   │   └── requirements.txt          # Module dependencies
│   │   ├── text_classification/
│   │   │   ├── data_preprocessing.py     # Text preprocessing
│   │   │   ├── rnn_model.py              # RNN/LSTM model
│   │   │   ├── transformer_model.py      # Transformer model
│   │   │   ├── train.py                  # Training script
│   │   │   ├── evaluate.py               # Evaluation script
│   │   │   └── requirements.txt          # Module dependencies
│   │   ├── multimodal_classification/
│   │   │   ├── data_preprocessing.py     # Multimodal data handling
│   │   │   ├── zero_shot_model.py        # Zero-shot learning
│   │   │   ├── few_shot_model.py         # Few-shot learning
│   │   │   ├── train.py                  # Training script
│   │   │   ├── evaluate.py               # Evaluation script
│   │   │   └── requirements.txt          # Module dependencies
│   └── .streamlit/
│       └── config.toml                    # Streamlit configuration
├── reports/                               # Documentation & analysis
│   ├── dataset_exploration.md             # Data analysis report
│   ├── data_loader_augmentation.md        # Data preparation report
│   ├── model_building_training.md         # Model architecture details
│   ├── experimental_results.md            # Results and comparisons
│   ├── extensions.md                      # Advanced techniques
│   └── final_report.md                    # Comprehensive final report
└── slides/                                # Presentation materials
    └── presentation.pptx                  # Presentation slides
```

## 🎯 How to Use Each Module

### Image Classification

```bash
cd github-pages/code/image_classification

# Training
python train.py

# Evaluation
python evaluate.py
```

### Text Classification

```bash
cd github-pages/code/text_classification

# Training
python train.py

# Evaluation
python evaluate.py
```

### Multimodal Classification

```bash
cd github-pages/code/multimodal_classification

# Training
python train.py

# Evaluation
python evaluate.py
```

## 📊 Key Features

### Data Processing
- Image augmentation (rotation, color jitter, random crop)
- Text tokenization and padding
- Multimodal data handling (image-text pairs)

### Models
- Custom PyTorch implementations
- Attention mechanisms
- Support for bidirectional processing
- Transformer architectures

### Training
- Early stopping with patience
- Learning rate scheduling
- Checkpoint management
- Loss and metric tracking

### Evaluation
- Classification metrics (accuracy, precision, recall, F1)
- Confusion matrices
- Per-class performance analysis
- Detailed reports

## 📝 Reports

Comprehensive reports are available in the `reports/` directory:

- **Dataset Exploration**: Data statistics and analysis
- **Data Loader & Augmentation**: Data preparation techniques
- **Model Building & Training**: Architecture details and training procedures
- **Experimental Results**: Performance comparison and analysis
- **Extensions**: Advanced techniques (ensemble, distillation, meta-learning)
- **Final Report**: Complete project summary and conclusions

## 🔧 Configuration

### Hyperparameters

Default configurations:
- Image size: 224x224
- Batch size: 32
- Learning rate: 1e-3
- Epochs: 100
- Early stopping patience: 15

Modify in respective `train.py` files.

### Streamlit Configuration

Settings are in `github-pages/.streamlit/config.toml`:
- Theme colors
- Display settings
- Performance options

## 📚 Documentation

- [Assignment 1](github-pages/assignment-1.md) - Detailed assignment requirements
- [Index](github-pages/index.md) - Project overview and links
- [Reports](reports/) - Detailed analysis and findings

## 🛠️ Development

### Code Structure
- Modular design with separate preprocessing, model, and training files
- Type hints for better code clarity
- Comprehensive docstrings

### Testing
```bash
# Test imports
python -c "from github-pages.code.image_classification.cnn_model import create_cnn_model; print('OK')"
```

### Adding New Models
1. Create model file in appropriate directory
2. Implement `nn.Module` subclass
3. Add `create_*_model()` factory function
4. Update training/evaluation scripts

## 📖 Learning Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Vision Transformers Paper](https://arxiv.org/abs/2010.11929)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

## 🎓 Course Information

- **Course**: Deep Learning
- **Assignment**: Assignment 1
- **Status**: In Progress
- **Last Updated**: March 2026

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 👥 Contributing

Contributions are welcome! Feel free to:
- Report issues
- Suggest improvements
- Submit pull requests
- Add new models or techniques

## 📞 Support

For questions or issues:
1. Check the [reports](reports/) directory
2. Review example code in respective modules
3. Check [assignment-1.md](github-pages/assignment-1.md)

---

**Project Start**: March 2026
**Virtual Environment**: Python 3.9.6
**Status**: Repository setup complete ✅