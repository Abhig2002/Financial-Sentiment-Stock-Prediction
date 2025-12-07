from typing import Optional
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_score, recall_score, f1_score

# Add parent directory to path to import embeddings module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from embeddings import get_embedder


class SVM:
    """
    Binary (0/1) text classifier using SentenceTransformer embeddings + LinearSVC.
    Expects dataframes with columns: 'Summary' (str), 'Truth' (0 or 1).
    """

    def __init__(self, embedding_backend: str = "finbert", random_state: int = 42):
        self.embedding_backend = embedding_backend
        self.encoder = get_embedder(model_name=embedding_backend)
        self.random_state = random_state
        # LinearSVC is fast and works well on dense embeddings.
        self.model = LinearSVC(random_state=random_state, dual="auto")
        self._is_fitted = False
        self.training_history = {
            "epoch": [],
            "precision": [],
            "recall": [],
            "f1": [],
        }

    @staticmethod
    def _ensure_binary(y: pd.Series) -> np.ndarray:
        y_arr = y.astype(int).to_numpy()
        uniq = set(np.unique(y_arr).tolist())
        if not uniq.issubset({0, 1}):
            raise ValueError(
                f"'Truth' must be binary 0/1. Found labels: {sorted(uniq)}"
            )
        return y_arr

    def _embed(self, texts: pd.Series) -> np.ndarray:
        # normalize_embeddings=True often helps linear models
        return self.encoder.encode(
            texts.astype(str).tolist(),
            batch_size=32,
            show_progress=False,
            normalize_embeddings=True,
        )

    def train(
        self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None
    ) -> None:
        """
        Train the Linear SVM classifier with progressive validation monitoring.
        
        Args:
            train_df: Training DataFrame with 'Summary' and 'Truth' columns
            val_df: Optional validation DataFrame for monitoring (not used in training)
        """
        # Only use train_df for training (avoid data leakage)
        df = train_df.copy()

        if "Summary" not in df or "Truth" not in df:
            raise KeyError("Dataframes must contain 'Summary' and 'Truth' columns.")

        if len(df) == 0:
            raise ValueError("Training dataframe is empty.")

        print(f"Training Linear SVM on {len(df)} samples...")
        
        # Embed training data once
        X_train = self._embed(df["Summary"])
        y_train = self._ensure_binary(df["Truth"])
        
        # Embed validation data if provided
        X_val, y_val = None, None
        if val_df is not None:
            if "Summary" not in val_df or "Truth" not in val_df:
                raise KeyError("val_df must contain 'Summary' and 'Truth' columns.")
            X_val = self._embed(val_df["Summary"])
            y_val = self._ensure_binary(val_df["Truth"])
            print(f"Validation set: {len(val_df)} samples")
        
        # Progressive training: train on increasing subsets (for visualization)
        if val_df is not None:
            n_epochs = 5  # Reduced from 10 for faster training
            for epoch in range(1, n_epochs + 1):
                # Use progressively more training data
                subset_size = int(len(X_train) * (epoch / n_epochs))
                subset_size = max(subset_size, 100)  # Minimum 100 samples
                
                X_subset = X_train[:subset_size]
                y_subset = y_train[:subset_size]
                
                # Train fresh model on subset
                self.model = LinearSVC(random_state=self.random_state, dual="auto", max_iter=5000)
                self.model.fit(X_subset, y_subset)
                
                # Evaluate on validation set
                y_pred = self.model.predict(X_val)
                
                precision = precision_score(y_val, y_pred, zero_division=0)
                recall = recall_score(y_val, y_pred, zero_division=0)
                f1 = f1_score(y_val, y_pred, zero_division=0)
                
                self.training_history["epoch"].append(epoch)
                self.training_history["precision"].append(precision)
                self.training_history["recall"].append(recall)
                self.training_history["f1"].append(f1)
                
                print(f"Epoch {epoch}/{n_epochs} ({subset_size} samples) - "
                      f"Val Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        # Final training on full dataset
        self.model = LinearSVC(random_state=self.random_state, dual="auto", max_iter=5000)
        self.model.fit(X_train, y_train)
        self._is_fitted = True
        print("Linear SVM training complete.")
        
        # Save training graph if validation was used
        if val_df is not None and len(self.training_history["epoch"]) > 0:
            self._save_training_graph()

    def predict(self, eval_df: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted. Call train() first.")
        
        if "Summary" not in eval_df or "Truth" not in eval_df:
            raise KeyError("eval_df must contain 'Summary' and 'Truth' columns.")

        if len(eval_df) == 0:
            raise ValueError("Evaluation dataframe is empty.")

        X = self._embed(eval_df["Summary"])
        preds = self.model.predict(X)

        # Ground truth (validated as 0/1)
        y_true = self._ensure_binary(eval_df["Truth"])

        return pd.DataFrame(
            {
                "Prediction": preds.astype(int),
                "Truth": y_true.astype(int),
            },
            index=eval_df.index,
        )

    def decision_scores(self, eval_df: pd.DataFrame) -> np.ndarray:
        """
        Returns signed distance to the decision boundary for each row
        (useful as a confidence score; larger magnitude = more confident).
        """
        if "Summary" not in eval_df:
            raise KeyError("eval_df must contain 'Summary' column.")
        X = self._embed(eval_df["Summary"])
        return self.model.decision_function(X)

    def _save_training_graph(self) -> None:
        """
        Save training history graph showing validation metrics per epoch.
        Graph is saved to ./output/svm/training_history.png
        """
        output_dir = os.path.join("output", "svm")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "training_history.png")
        
        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle("Linear SVM Training: Validation Performance per Epoch", fontsize=14, fontweight="bold")
        
        epochs = self.training_history["epoch"]
        
        # Precision plot
        axes[0].plot(epochs, self.training_history["precision"], marker='o', linewidth=2, color='#2E86AB')
        axes[0].set_xlabel("Epoch", fontsize=11)
        axes[0].set_ylabel("Precision", fontsize=11)
        axes[0].set_title("Validation Precision", fontsize=12, fontweight="bold")
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim([0, 1.05])
        
        # Recall plot
        axes[1].plot(epochs, self.training_history["recall"], marker='s', linewidth=2, color='#A23B72')
        axes[1].set_xlabel("Epoch", fontsize=11)
        axes[1].set_ylabel("Recall", fontsize=11)
        axes[1].set_title("Validation Recall", fontsize=12, fontweight="bold")
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 1.05])
        
        # F1 Score plot
        axes[2].plot(epochs, self.training_history["f1"], marker='^', linewidth=2, color='#F18F01')
        axes[2].set_xlabel("Epoch", fontsize=11)
        axes[2].set_ylabel("F1 Score", fontsize=11)
        axes[2].set_title("Validation F1 Score", fontsize=12, fontweight="bold")
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylim([0, 1.05])
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Saved training history graph to {output_path}")
