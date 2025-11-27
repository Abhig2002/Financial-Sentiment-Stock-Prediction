
# RBF SVM Extension for the Stock Movement Classification Pipeline

This markdown file introduces an **RBF-kernel Support Vector Machine module** designed to extend the existing `SVM` class used in the stock price prediction pipeline.  
It integrates directly with the current MiniLM embedding workflow and shares the same interface as the original `SVM` classifier.

---

## 📌 Purpose

The current pipeline uses:

- SentenceTransformer embeddings (`all-MiniLM-L6-v2`)
- A linear Support Vector Classifier (`LinearSVC`)

This document adds an **RBF-kernel SVM** (`sklearn.svm.SVC`) so the pipeline can:

- Model **non-linear decision boundaries**
- Run hyperparameter sweeps over `C` and `gamma`
- Compare linear vs. non-linear SVM performance
- Support portfolio requirements (extra experiments)

The RBF SVM **inherits** from the original `SVM` class and overrides only the classifier component.  
All preprocessing, embedding, training, and evaluation workflows remain the same.

---

## 📁 New File: `svm_rbf.py`

Place this file in the same directory as your existing SVM implementation (e.g., `/models`).

```python
from sklearn.svm import SVC
from typing import Optional
from .svm import SVM   # adjust import if your path differs


class SVM_RBF(SVM):
    """
    RBF-kernel SVM for stock movement prediction.

    Inherits:
        - SBERT embedding pipeline
        - train() and predict() workflow
        - label validation utilities

    Overrides:
        - self.model (uses SVC with an RBF kernel)
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        C: float = 1.0,
        gamma: str | float = "scale",
        random_state: int = 42,
    ):
        # Initialize parent SBERT encoder
        super().__init__(model_name=model_name, random_state=random_state)

        # Override LinearSVC with RBF-kernel SVC
        self.model = SVC(
            kernel="rbf",
            C=C,
            gamma=gamma,
            probability=False,
            random_state=random_state,
        )

        # Store hyperparameters for logging/debugging
        self.C = C
        self.gamma = gamma

    def __repr__(self):
        return f"SVM_RBF(C={{self.C}}, gamma={{self.gamma}})"
```

---

## 🧪 Example Usage

```python
import pandas as pd
from models.svm_rbf import SVM_RBF

train_df = pd.read_csv("train.csv")
val_df = pd.read_csv("val.csv")

rbf = SVM_RBF(C=1.0, gamma="scale")
rbf.train(train_df, val_df)

results = rbf.predict(val_df)
print(results.head())
```

---

## 🔄 Hyperparameter Sweep Example

```python
C_values = [0.1, 1, 10]
gamma_values = ["scale", "auto"]

records = []

for c in C_values:
    for g in gamma_values:
        clf = SVM_RBF(C=c, gamma=g)
        clf.train(train_df, val_df)
        preds = clf.predict(val_df)

        f1 = f1_score(preds["Truth"], preds["Prediction"])
        records.append((c, g, f1))

print(records)
```

---

## ✔️ Benefits of Adding RBF SVM

- Captures non-linear structures in embedding space  
- Good complement to LinearSVC baseline  
- Enables extra experimentation  
- Compatible with all existing pipeline components  
- Minimal code changes — plug-and-play inheritance

---

## 📌 Notes

- RBF models are slower than LinearSVC on large datasets.  
- Recommended for:
  - hyperparameter sweeps  
  - smaller subsets  
  - portfolio experiments  
  - ablation studies  

---

## ✅ Summary

This extension introduces a fully compatible **RBF-kernel SVM** to your system.  
To activate the RBF SVM:

```python
model = SVM_RBF(C=1.0, gamma="scale")
```

---

# End of File
