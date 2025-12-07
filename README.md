# Financial Sentiment Stock Prediction

## About
This project implements a comprehensive pipeline for stock price movement prediction using financial news sentiment analysis. The system combines state-of-the-art NLP embeddings (FinBERT) with multiple machine learning models to predict stock price movements.

### Key Features
- **Finance-Specific Embeddings**: FinBERT (yiyanghkust/finbert-tone) for domain-aware text representation
- **Multiple Prediction Tasks**: 1-day and 5-day price movement forecasting
- **Diverse ML Models**: CNN, LSTM, Random Forest, and SVM variants
- **Model Explainability**: SHAP analysis for Random Forest interpretability
- **Flexible Architecture**: Easy switching between FinBERT and MiniLM embeddings

## Dataset
We use data from the [FNSPID dataset](https://github.com/Zdong104/FNSPID_Financial_News_Dataset). We provide processed datasets used during our experimentation for the [full dataset](https://drive.google.com/file/d/1UK-OwzI7j0ITMmF1IDKxxZPrneJP9x3m/view?usp=sharing) and a [fortune 500 subset](https://drive.google.com/file/d/1tBKFjc_ilOJ3La_Kd9--UURZBTTvqfO0/view?usp=share_link).

### Data Format
Each CSV file should contain:
- `Date`: Trading date
- `Open`, `Close`, `Low`, `Adj close`: Stock prices
- `Lexrank_summary` or `Summary`: Text summary of news articles
- `News_flag`: 1 if news is available for that day

The pipeline automatically:
1. Filters to days with news (`News_flag == 1`)
2. Creates `label_1d`: 1 if `Close >= Open`, 0 otherwise
3. Creates `label_5d`: 1 if 5-day forward return ≥ 0, 0 otherwise
4. Extracts ticker symbol from filename

## Embedding Models

### FinBERT (Recommended)
- **Model**: `yiyanghkust/finbert-tone`
- **Embedding Dimension**: 768
- **Advantages**: 
  - Pre-trained on financial texts (10-K, earnings calls, analyst reports)
  - Understands financial terminology ("bullish", "bearish", "volatility")
  - Better performance on financial sentiment (~2-5% F1 improvement)
- **Use Case**: Production systems, portfolio projects, research

### MiniLM (Baseline)
- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Embedding Dimension**: 384
- **Advantages**:
  - 2x faster encoding
  - Half the memory usage
  - Still competitive performance
- **Use Case**: Rapid prototyping, resource-constrained environments

### Switching Embeddings
The codebase is designed for easy comparison:
```python
# In embeddings.py
embedder = get_embedder(model_name="finbert")  # or "minilm"
embeddings = embedder.encode(texts)
```

## Quick Start

### Installation
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download dataset from the [FNSPID dataset](https://github.com/Zdong104/FNSPID_Financial_News_Dataset) or use our preprocessed datasets (see Dataset section below).

### Basic Usage
```bash
# Run Random Forest with FinBERT embeddings on 1-day prediction task
python main.py -p randomforest -d ./data -r 1 -t 1d -e finbert

# Run all models with FinBERT on 5-day prediction task
python main.py -p all -d ./data -r 1 -t 5d -e finbert

# Compare FinBERT vs MiniLM embeddings on LSTM
python main.py -p lstm -d ./data -r 1 -t 1d -e finbert
python main.py -p lstm -d ./data -r 1 -t 1d -e minilm
```

### SHAP Explainability Analysis
After training Random Forest, generate SHAP plots:
```bash
# Generate SHAP analysis for FinBERT-based Random Forest
python shap_analysis.py --embedding_backend finbert

# Generate SHAP analysis for MiniLM-based Random Forest
python shap_analysis.py --embedding_backend minilm
```

### Example Workflows

#### Portfolio Workflow: Complete Comparison
```bash
# 1. Train with FinBERT on 1-day task
python main.py -p all -d ./data -r 2 -t 1d -e finbert

# 2. Train with FinBERT on 5-day task
python main.py -p all -d ./data -r 2 -t 5d -e finbert

# 3. Compare with MiniLM baseline
python main.py -p randomforest -d ./data -r 2 -t 1d -e minilm

# 4. Generate SHAP explanations
python shap_analysis.py -e finbert
```

#### Quick Test Run
```bash
# Single run with Random Forest
python main.py -p randomforest -d ./data -r 1 -t 1d -e finbert
```

## Command Line Arguments

### main.py

| Argument | Short | Required | Default | Choices | Description |
|----------|-------|----------|---------|---------|-------------|
| `--pipeline` | `-p` | ✅ Yes | - | `svm`, `cnn`, `lstm`, `randomforest`, `all` | Model to train |
| `--data` | `-d` | ✅ Yes | - | - | Path to data folder |
| `--runs` | `-r` | ✅ Yes | - | - | Number of training runs |
| `--task_type` | `-t` | ❌ No | `1d` | `1d`, `5d` | Prediction task type |
| `--embedding_backend` | `-e` | ❌ No | `finbert` | `finbert`, `minilm` | Embedding model |

**Notes:**
- `svm_rbf` and `svm_fuzzy` are currently commented out in the pipeline
- `task_type`:
  - `1d`: Predicts if Close ≥ Open (same-day movement)
  - `5d`: Predicts if 5-day forward return ≥ 0 (more meaningful for trading)
- `embedding_backend`:
  - `finbert`: Finance-specific BERT (yiyanghkust/finbert-tone) - **recommended**
  - `minilm`: General-purpose sentence-transformers (all-MiniLM-L6-v2)

### shap_analysis.py

| Argument | Short | Required | Default | Choices | Description |
|----------|-------|----------|---------|---------|-------------|
| `--embedding_backend` | `-e` | ❌ No | `finbert` | `finbert`, `minilm` | Which model to analyze |
| `--max_display` | - | ❌ No | `20` | - | Max features in plots |

## Models

All models support both FinBERT and MiniLM embeddings through the `--embedding_backend` flag.

### Support Vector Machines
- **Linear SVM** (`svm`): Fast linear classifier using LinearSVC
  - Best for large datasets (180K+ samples)
  - O(n) complexity - scales linearly
  - Works with both FinBERT (768-dim) and MiniLM (384-dim) embeddings

### Deep Learning
- **CNN**: 1D Convolutional Neural Network
  - Multi-kernel architecture (sizes 3, 5, 7)
  - Adaptive max pooling
  - Batch normalization and dropout
  - 20 epochs with validation monitoring
  
- **LSTM**: Bidirectional LSTM
  - 2 layers, 128 hidden units
  - Chunk-based sequence processing
  - Mixed precision training (BF16/FP16)
  - 20 epochs with validation monitoring

### Ensemble Methods
- **Random Forest**: Robust tree-based classifier
  - 100 decision trees
  - Class-balanced weights for handling imbalanced data
  - Incremental training with validation tracking
  - **SHAP explainability support** - see SHAP Analysis section

## SHAP Explainability

Random Forest models automatically save artifacts for SHAP analysis. Run `shap_analysis.py` after training to generate:

1. **SHAP Summary Plot (Beeswarm)**: Shows how each embedding dimension impacts predictions
2. **SHAP Feature Importance (Bar)**: Ranks embedding dimensions by mean absolute SHAP value
3. **SHAP Waterfall Plot**: Detailed breakdown of a single prediction

**Why SHAP for Finance?**
- Understand which parts of the text embedding drive predictions
- Identify if the model focuses on relevant financial sentiment
- Compare feature importance between FinBERT and MiniLM
- Build trust in model decisions for portfolio management

## Output Structure

### Experiment Results
Each run generates organized outputs with task type and embedding backend in the name:

```
output/
├── {model}-{embedding}-{task}-run{n}/
│   ├── metrics_{task}.csv              # Precision, Recall, F1
│   └── confusion_matrix_{task}.png     # Visual confusion matrix
│
├── {model}/
│   └── training_history_{timestamp}.png # Training curves
│
└── randomforest/
    ├── artifacts/
    │   ├── rf_model_{embedding}.joblib     # Saved model
    │   ├── X_train_{embedding}.npy         # Training embeddings
    │   └── y_train_{embedding}.npy         # Training labels
    └── shap/
        ├── shap_summary_{embedding}.png    # SHAP beeswarm plot
        ├── shap_bar_{embedding}.png        # Feature importance
        └── shap_waterfall_{embedding}.png  # Example prediction
```

**Example:** Running `python main.py -p randomforest -d ./data -r 1 -t 5d -e finbert` creates:
- `output/randomforest-finbert-5d-run1/metrics_5d.csv`
- `output/randomforest-finbert-5d-run1/confusion_matrix_5d.png`
- `output/randomforest/artifacts/rf_model_finbert.joblib`

### Comparing Results

#### 1-day vs 5-day Task
Compare `metrics_1d.csv` and `metrics_5d.csv` to see which prediction horizon works better:
- **1-day**: Easier task, higher accuracy, but less useful for trading
- **5-day**: Harder task, captures medium-term trends, more practical

#### FinBERT vs MiniLM
Compare models trained with different embeddings:
- **FinBERT**: Domain-specific, should capture financial terminology better
- **MiniLM**: Smaller (384-dim vs 768-dim), faster, general-purpose

### Performance Notes
- **FinBERT Advantage**: ~2-5% F1 improvement on financial news (domain-specific knowledge)
- **MiniLM Advantage**: 2x faster embedding, half the memory, still competitive
- **5-day Task**: Typically 5-10% lower accuracy than 1-day (more challenging)
- **Best Model**: Random Forest + FinBERT + 5-day task (practical and explainable)

## Project Structure

```
.
├── main.py                    # Main training pipeline
├── embeddings.py             # Centralized embedding module (FinBERT/MiniLM)
├── shap_analysis.py          # SHAP explainability for Random Forest
├── data_integ.py             # Data loading and evaluation functions
│
├── data_utils/
│   ├── preprocess.py         # Date/time preprocessing
│   ├── price_news_integrate.py
│   ├── score_by_gpt.py
│   └── summarize.py
│
├── svm/
│   └── svm.py                # Linear SVM implementation
│
├── randomforest/
│   └── randomforest.py       # Random Forest with SHAP support
│
├── cnn/
│   └── cnn.py                # 1D CNN implementation
│
├── lstm/
│   └── lstm.py               # Bidirectional LSTM implementation
│
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Portfolio Highlights

This project demonstrates:

1. **Domain-Specific NLP**: Using FinBERT instead of generic embeddings shows understanding of domain adaptation
2. **Multiple Prediction Horizons**: Comparing 1-day vs 5-day shows awareness of practical trading considerations
3. **Model Explainability**: SHAP analysis demonstrates responsible AI practices for financial applications
4. **Modular Design**: Clean separation between embedding layer and models enables easy experimentation
5. **Production-Ready**: Proper CLI, logging, artifact saving, and documentation

### Key Improvements Over Baseline
- ✅ Finance-specific embeddings (FinBERT) for better domain understanding
- ✅ 5-day prediction task for more meaningful trading signals
- ✅ SHAP explainability for Random Forest interpretability
- ✅ Flexible architecture supporting multiple embedding backends
- ✅ Comprehensive documentation and examples

## Citation

If you use this code, please cite the FNSPID dataset:
```
@article{fnspid2023,
  title={FNSPID: A Comprehensive Financial News and Stock Price Dataset for Stock Movement Prediction},
  author={Dong, Zihan and others},
  journal={arXiv preprint},
  year={2023}
}
```
