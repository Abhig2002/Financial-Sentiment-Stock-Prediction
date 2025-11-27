from typing import Optional
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_score, recall_score, f1_score
from .svm import SVM


class SVM_Fuzzy(SVM):
    """
    Fuzzy Support Vector Machine for binary (0/1) text classification.
    
    Implements Fuzzy Linear SVM by providing sample weights to LinearSVC.
    Memberships are computed using distance from class centroids, making
    the model more robust to noisy/outlier financial news.
    
    Reference:
        Lin, C.-F., and Wang, S.-D., "Fuzzy Support Vector Machines,"
        IEEE Transactions on Neural Networks, vol. 13, no. 2, 2002.
    
    Inherits from SVM class:
        - SentenceTransformer embedding pipeline
        - Label validation utilities
    
    Key difference:
        - Assigns fuzzy membership weights to samples during training
        - Reduces influence of outliers and noisy samples
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        min_membership: float = 0.1,
        random_state: int = 42,
    ):
        """
        Initialize the Fuzzy SVM classifier with SentenceTransformer embeddings.
        
        Args:
            model_name: Name of the SentenceTransformer model for text embeddings
            min_membership: Minimum membership value (prevents zero weights)
            random_state: Random seed for reproducibility
        """
        # Initialize parent class (gets encoder and sets up embedding pipeline)
        super().__init__(model_name=model_name, random_state=random_state)

        # Store hyperparameters
        self.min_membership = min_membership
        self._is_fitted = False
        self.training_history = {
            "epoch": [],
            "precision": [],
            "recall": [],
            "f1": [],
        }
        self.memberships_ = None  # Store computed memberships for analysis

    @staticmethod
    def _compute_fuzzy_memberships(
        X: np.ndarray, y: np.ndarray, min_membership: float = 0.1
    ) -> np.ndarray:
        """
        Compute fuzzy membership values for each training sample.
        
        Membership is based on distance from class centroid:
        - Samples close to their class center get higher membership (≈ 1.0)
        - Samples far from their class center get lower membership (≈ min_membership)
        - This downweights outliers/noisy samples
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (n_samples,)
            min_membership: Minimum membership value (0 < min_membership ≤ 1)
            
        Returns:
            Array of membership values (n_samples,)
        """
        # Separate samples by class
        X0 = X[y == 0]
        X1 = X[y == 1]

        # Compute class centroids
        mu0 = X0.mean(axis=0)
        mu1 = X1.mean(axis=0)

        # Compute distance from each sample to its class centroid
        dists = np.zeros(len(y))
        for i, label in enumerate(y):
            center = mu0 if label == 0 else mu1
            dists[i] = np.linalg.norm(X[i] - center)

        # Normalize distances to [0, 1]
        dmax = dists.max()
        if dmax == 0:
            # All samples at centroid (unlikely but handle it)
            return np.ones_like(dists)

        # Convert distance to membership: far = low membership
        memberships = 1.0 - (dists / dmax)
        
        # Scale to [min_membership, 1.0]
        memberships = min_membership + (1.0 - min_membership) * memberships
        
        return memberships

    def train(
        self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None
    ) -> None:
        """
        Train the Fuzzy SVM classifier with fuzzy membership weights.
        
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

        print(f"Training Fuzzy SVM (min_membership={self.min_membership}) on {len(df)} samples...")
        
        # Embed training data once
        X_train = self._embed(df["Summary"])
        y_train = self._ensure_binary(df["Truth"])
        
        # Compute fuzzy memberships
        print("Computing fuzzy memberships...")
        memberships = self._compute_fuzzy_memberships(X_train, y_train, self.min_membership)
        self.memberships_ = memberships  # Store for analysis
        
        # Print membership statistics
        print(f"Membership stats: min={memberships.min():.3f}, "
              f"max={memberships.max():.3f}, mean={memberships.mean():.3f}")
        
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
            n_epochs = 5
            for epoch in range(1, n_epochs + 1):
                # Use progressively more training data
                subset_size = int(len(X_train) * (epoch / n_epochs))
                subset_size = max(subset_size, 100)  # Minimum 100 samples
                
                X_subset = X_train[:subset_size]
                y_subset = y_train[:subset_size]
                memberships_subset = memberships[:subset_size]
                
                # Train fresh model on subset with fuzzy weights
                self.model = LinearSVC(random_state=self.random_state, dual="auto", max_iter=5000)
                self.model.fit(X_subset, y_subset, sample_weight=memberships_subset)
                
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
        
        # Final training on full dataset with fuzzy weights
        self.model = LinearSVC(random_state=self.random_state, dual="auto", max_iter=5000)
        self.model.fit(X_train, y_train, sample_weight=memberships)
        self._is_fitted = True
        print("Fuzzy SVM training complete.")
        
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

    def get_membership_distribution(self) -> dict:
        """
        Get statistics about the fuzzy memberships computed during training.
        
        Returns:
            Dictionary with membership statistics
        """
        if self.memberships_ is None:
            raise RuntimeError("Model must be trained first to access memberships.")
        
        return {
            "min": float(self.memberships_.min()),
            "max": float(self.memberships_.max()),
            "mean": float(self.memberships_.mean()),
            "std": float(self.memberships_.std()),
            "median": float(np.median(self.memberships_)),
        }

    def _save_training_graph(self) -> None:
        """
        Save training history graph showing validation metrics per epoch.
        Graph is saved to ./output/svm_fuzzy/training_history.png
        """
        output_dir = os.path.join("output", "svm_fuzzy")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "training_history.png")
        
        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(f"Fuzzy SVM Training (min_membership={self.min_membership}): Validation Performance per Epoch", 
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
        return f"SVM_Fuzzy(min_membership={self.min_membership})"

