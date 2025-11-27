
# Fuzzy SVM Extension for Stock Movement Prediction Pipeline

This README introduces a **Fuzzy Support Vector Machine (FSVM)** module that extends the existing
`SVM` pipeline used for stock movement prediction.  
It provides fuzzy membership weights to improve robustness against noisy financial news,
while maintaining **full scalability** to large datasets (180k+ summaries).

---

## 📌 1. What Is Fuzzy SVM?

A **Fuzzy Support Vector Machine (FSVM)** assigns a membership value `s_i ∈ (0,1]` to each training sample.
These memberships weight the hinge loss, reducing the influence of outliers or noisy samples.

This method comes from:

Lin, C.-F., and Wang, S.-D., "Fuzzy Support Vector Machines," IEEE Transactions on Neural Networks, 2002.

---

## 📁 2. New File: svm_fuzzy.py

Place this file next to your existing svm.py.

```
# FILE: svm_fuzzy.py

import numpy as np
from sklearn.svm import LinearSVC
from typing import Optional
import pandas as pd
from .svm import SVM   # adjust import path if needed


class SVM_Fuzzy(SVM):
    """
    Implements Fuzzy Linear SVM by providing sample weights to LinearSVC.

    Memberships are computed using distance from class centroids.
    This makes the model more robust to noisy/outlier financial news.
    """

    @staticmethod
    def _compute_fuzzy_memberships(X: np.ndarray, y: np.ndarray,
                                   min_membership: float = 0.1) -> np.ndarray:
        # Separate by class
        X0 = X[y == 0]
        X1 = X[y == 1]

        mu0 = X0.mean(axis=0)
        mu1 = X1.mean(axis=0)

        dists = np.zeros(len(y))
        for i, label in enumerate(y):
            center = mu0 if label == 0 else mu1
            dists[i] = np.linalg.norm(X[i] - center)

        dmax = dists.max()
        if dmax == 0:
            return np.ones_like(dists)

        memberships = 1.0 - (dists / dmax)
        memberships = min_membership + (1 - min_membership) * memberships
        return memberships

    def train(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None,
              min_membership: float = 0.1):

        if val_df is not None:
            df = pd.concat([train_df, val_df], ignore_index=True)
        else:
            df = train_df.copy()

        X = self._embed(df["Summary"])
        y = self._ensure_binary(df["Truth"])

        memberships = self._compute_fuzzy_memberships(X, y, min_membership)
        self.model.fit(X, y, sample_weight=memberships)
```

---

## 🚀 3. Example Usage

```
from models.svm_fuzzy import SVM_Fuzzy
import pandas as pd

train_df = pd.read_csv("train.csv")
val_df = pd.read_csv("val.csv")

model = SVM_Fuzzy()
model.train(train_df, val_df)

preds = model.predict(val_df)
print(preds.head())
```

---

## 📊 4. Integration Notes

- FSVM is **as fast as LinearSVC** (unlike RBF SVM which is O(N²)).  
- Works with **full 180k sample dataset**.  
- Downweights noisy news articles automatically.

---

## 📚 Reference (IEEE)

Lin, C.-F., and Wang, S.-D., "Fuzzy Support Vector Machines," *IEEE Transactions on Neural Networks*, vol. 13, no. 2, 2002.

---

# ✅ End of File
