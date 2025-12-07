# Portfolio Upgrade Instructions — Stock Price Prediction with Sentiment-Aware Embeddings

This README tells the coding assistant (Cursor) **exactly what code changes to make** to turn the existing class project into a **portfolio-ready version** in under a week.

The existing project already:
- Loads the FNSPID dataset
- Summarizes news text (LexRank/LSA)
- Embeds summaries using `sentence-transformers/all-MiniLM-L6-v2`
- Trains 4 models on 1-day labels (Close ≥ Open): SVM, Random Forest, CNN, LSTM
- Evaluates with Accuracy / Precision / Recall / F1

We now want to implement **three concrete upgrades**:

1. **Swap MiniLM sentence embeddings → FinBERT embeddings** (finance-specific)
2. **Add a second prediction task: 5-day price movement** (more meaningful target)
3. **Add SHAP explainability for the best traditional model (Random Forest)**

> ⚠️ Important: Do **not** delete the original 1-day task pipeline. We want both versions to coexist so we can compare performance.

---

## 0. Project Structure Assumptions

Assume the project has a structure roughly like:

- `data/` — raw & processed CSVs  
- `src/`
  - `data_preprocessing.py`
  - `embeddings.py`
  - `models/`
    - `svm_model.py`
    - `random_forest_model.py`
    - `cnn_model.py`
    - `lstm_model.py`
  - `train.py` or `run_experiments.py`
- `configs/` — hyperparameters
- `results/` — metrics, plots, etc.

If file names differ, apply the same logic to corresponding files.

---

## 1. Upgrade Embeddings: MiniLM → FinBERT

### 1.1. Dependencies

Add to `requirements.txt`:
- `transformers`
- `torch`
- `accelerate` (optional)

### 1.2. Create/Update `src/embeddings.py`

Provide a unified embedding function:

```python
def get_text_embeddings(texts, model_name="finbert", device=None):
    ...
```

#### Implement FinBERT logic

Use: `"yiyanghkust/finbert-tone"`

Requirements:
- Use AutoTokenizer + AutoModel
- GPU if available
- Batch processing
- Mean pooling with attention mask

(Cursor will convert pseudocode into working code.)

### 1.3. Replace all `SentenceTransformer("all-MiniLM-L6-v2")` usages

Refactor to:

```python
from src.embeddings import get_text_embeddings
embs = get_text_embeddings(list_of_summaries, model_name="finbert")
```

Do NOT modify downstream model code.

---

## 2. New Task: 5-Day Price Movement Label

### 2.1. Modify `data_preprocessing.py`

#### Explicit 1-day label

```python
df["label_1d"] = (df["Close"] >= df["Open"]).astype(int)
```

#### Create 5-day label

```python
df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
df["Close_t_plus_5"] = df.groupby("Ticker")["Close"].shift(-5)
df["ret_5d"] = (df["Close_t_plus_5"] - df["Close"]) / df["Close"]
df = df.dropna(subset=["Close_t_plus_5"])
df["label_5d"] = (df["ret_5d"] >= 0).astype(int)
```

Keep both labels in the final dataset.

### 2.2. Add CLI flag in training scripts

```python
--task_type {1d,5d}
```

Label selection:

```python
y = df["label_1d"] if args.task_type=="1d" else df["label_5d"]
```

### 2.3. Save metrics separately

Examples:
- `results/svm_metrics_1d.json`
- `results/svm_metrics_5d.json`

---

## 3. SHAP Explainability for Random Forest

### 3.1. Dependencies

Add to `requirements.txt`:  
- `shap`
- `matplotlib`

### 3.2. Save RF model + embeddings

After training RF, save artifacts:

```python
joblib.dump(rf_model, "rf_model_{task}.joblib")
np.save("X_train_{task}.npy", X_train)
np.save("y_train_{task}.npy", y_train)
```

### 3.3. Create `src/shap_analysis.py`

Load saved model + embeddings and generate:

- SHAP summary plot  
- SHAP bar plot

Output to:
```
results/shap/
```

---

## 4. How to Run the Upgraded Pipeline

### 4.1. Add CLI flag

```python
--embedding_backend {finbert,minilm}
```

Default: `"finbert"`.

### 4.2. Example commands

```bash
python src/train.py --embedding_backend finbert --task_type 1d
python src/train.py --embedding_backend finbert --task_type 5d
python src/shap_analysis.py
```

---

## 5. Portfolio Documentation Notes

After implementing:

- Describe FinBERT upgrade
- Describe 5-day task motivation
- Show 1-day vs 5-day comparison
- Include SHAP plots  

Cursor should ensure:
- Clean, modular code
- Clear artifact directory structure
- Updates to README explaining new flags

This completes the required portfolio-ready upgrades.
