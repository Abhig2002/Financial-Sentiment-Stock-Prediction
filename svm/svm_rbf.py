from typing import Optional, Union
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score
from .svm import SVM


class SVM_RBF(SVM):
    """
    RBF-kernel SVM for binary (0/1) text classification.
    
    Inherits from SVM class:
        - SentenceTransformer embedding pipeline
        - train() and predict() workflow
        - Label validation utilities
    
    Overrides:
        - self.model (uses SVC with RBF kernel instead of LinearSVC)
    
    Captures non-linear decision boundaries in the embedding space.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        C: float = 1.0,
        gamma: Union[str, float] = "scale",
        random_state: int = 42,
    ):
        """
        Initialize the RBF-kernel SVM classifier with SentenceTransformer embeddings.
        
        Args:
            model_name: Name of the SentenceTransformer model for text embeddings
            C: Regularization parameter (higher = less regularization)
            gamma: Kernel coefficient ('scale', 'auto', or float value)
            random_state: Random seed for reproducibility
        """
        # Initialize parent class (gets encoder and sets up embedding pipeline)
        super().__init__(model_name=model_name, random_state=random_state)

        # Override LinearSVC with RBF-kernel SVC
        self.model = SVC(
            kernel="rbf",
            C=C,
            gamma=gamma,
            probability=False,  # Faster without probability estimates
            random_state=random_state,
        )

        # Store hyperparameters for logging/debugging
        self.C = C
        self.gamma = gamma
        self._is_fitted = False
        self.training_history = {
            "epoch": [],
            "precision": [],
            "recall": [],
            "f1": [],
        }

    def train(
        self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None
    ) -> None:
        """
        Train the RBF SVM classifier with progressive validation monitoring.
        
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

        print(f"Training RBF SVM (C={self.C}, gamma={self.gamma}) on {len(df)} samples...")
        
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
        
        # Progressive training: train on increasing subsets
        n_epochs = 10
        for epoch in range(1, n_epochs + 1):
            # Use progressively more training data
            subset_size = int(len(X_train) * (epoch / n_epochs))
            subset_size = max(subset_size, 100)  # Minimum 100 samples
            
            X_subset = X_train[:subset_size]
            y_subset = y_train[:subset_size]
            
            # Train fresh model on subset
            self.model = SVC(
                kernel="rbf",
                C=self.C,
                gamma=self.gamma,
                probability=False,
                random_state=self.random_state,
            )
            self.model.fit(X_subset, y_subset)
            
            # Evaluate on validation set if provided
            if val_df is not None:
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
        self.model = SVC(
            kernel="rbf",
            C=self.C,
            gamma=self.gamma,
            probability=False,
            random_state=self.random_state,
        )
        self.model.fit(X_train, y_train)
        self._is_fitted = True
        print("RBF SVM training complete.")
        
        # Save training graph if validation was used
        if val_df is not None and len(self.training_history["epoch"]) > 0:
            self._save_training_graph()

    def predict(self, eval_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions for the evaluation dataset.
        
        Args:
            eval_df: DataFrame with 'Summary' and 'Truth' columns
            
        Returns:
            DataFrame with 'Prediction' and 'Truth' columns
        """
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

    def decision_function(self, eval_df: pd.DataFrame) -> np.ndarray:
        """
        Returns signed distance to the decision boundary for each row.
        Larger magnitude = more confident prediction.
        
        Args:
            eval_df: DataFrame with 'Summary' column
            
        Returns:
            Array of decision function values
        """
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted. Call train() first.")
        
        if "Summary" not in eval_df:
            raise KeyError("eval_df must contain 'Summary' column.")
        
        X = self._embed(eval_df["Summary"])
        return self.model.decision_function(X)

    def _save_training_graph(self) -> None:
        """
        Save training history graph showing validation metrics per epoch.
        Graph is saved to ./output/svm_rbf/training_history.png
        """
        output_dir = os.path.join("output", "svm_rbf")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "training_history.png")
        
        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(f"RBF SVM Training (C={self.C}, gamma={self.gamma}): Validation Performance per Epoch", 
                     fontsize=14, fontweight="bold")
        
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

    def __repr__(self):
        return f"SVM_RBF(C={self.C}, gamma={self.gamma})"

