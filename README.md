# SGVT-FD: Semantic Grouped Visual Tokenization for Fault Diagnosis

Official PyTorch implementation of **"Semantic Grouped Visual Tokenization for Efficient VLM-based Fault Diagnosis"**.

## 🎯 Overview

SGVT-FD achieves **98.72% accuracy** on CRWU dataset with only **32 visual tokens** (6× reduction from 196), using just **2.2M parameters**.

**Key Features:**
- 🚀 6× token reduction (196 → 32)
- 📊 Competitive accuracy (98.72% CRWU, 98.13% MFPT)
- ⚡ Fast inference (0.31ms on RTX 5090)
- 💡 Interpretable semantic grouping

## 📦 Installation

```bash
# Clone repository
git clone https://github.com/yourusername/sgvt-fd.git
cd sgvt-fd

# Create environment
conda create -n sgvt python=3.8
conda activate sgvt

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Quick Start

### Training

```bash
python src/scripts/train_sgvt.py \
    --dataset crwu \
    --model sgvt_mi \
    --num_groups 32 \
    --batch_size 16 \
    --epochs 50
```

### Evaluation

```bash
python src/scripts/evaluate.py \
    --dataset crwu \
    --model_path results/best_model.pth \
    --num_groups 32
```

### Generate Figures

```bash
# Method overview figure
python generate_method_figure.py

# Results figures
python generate_results_figures.py
```

## 📊 Results

### Main Results on CRWU

| Method | Tokens | Params (M) | Accuracy (%) |
|--------|--------|------------|--------------|
| ViT Baseline | 196 | 43.5 | 98.96 ± 0.03 |
| CVT | 32 | 1.3 | 64.99 ± 1.29 |
| **SGVT-FD** | **32** | **2.2** | **98.72 ± 0.22** |

### Comparison with VLM Methods

| Method | Tokens | Accuracy (%) |
|--------|--------|--------------|
| BearLLM | 196 | 96.52 |
| FaultGPT | 196 | 93.17 |
| **SGVT-FD** | **32** | **98.72** |

## 📁 Project Structure

```
code-repo/
├── src/
│   ├── data/              # Data loaders (CRWU, MFPT)
│   ├── models/            # Model implementations
│   ├── scripts/           # Training & evaluation scripts
│   └── utils/             # Utilities (metrics, visualization)
├── generate_method_figure.py
├── generate_results_figures.py
├── run_*.py               # Experiment scripts
├── requirements.txt
└── README.md
```

## 🔧 Configuration

Edit `src/config.py` to customize:
- Dataset paths
- Model hyperparameters  
- Training settings

## 📄 License

Apache License 2.0

## 🙏 Acknowledgments

Built with [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) and [PyTorch](https://pytorch.org/).
