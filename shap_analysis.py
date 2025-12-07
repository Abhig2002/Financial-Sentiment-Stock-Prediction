"""
SHAP (SHapley Additive exPlanations) Analysis for Random Forest Model

This script loads a trained Random Forest model and generates SHAP plots
to explain model predictions and feature importance.

Usage:
    python shap_analysis.py --embedding_backend finbert
    python shap_analysis.py --embedding_backend minilm
"""

import argparse
import os
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import shap


def run_shap_analysis(embedding_backend: str = "finbert", max_display: int = 20):
    """
    Generate SHAP analysis plots for a trained Random Forest model.
    
    Args:
        embedding_backend: 'finbert' or 'minilm' - which model artifacts to load
        max_display: Maximum number of features to display in plots
    """
    print(f"🔍 Running SHAP analysis for Random Forest with {embedding_backend} embeddings...")
    
    # Define paths
    artifacts_dir = os.path.join("output", "randomforest", "artifacts")
    shap_dir = os.path.join("output", "randomforest", "shap")
    os.makedirs(shap_dir, exist_ok=True)
    
    model_path = os.path.join(artifacts_dir, f"rf_model_{embedding_backend}.joblib")
    X_path = os.path.join(artifacts_dir, f"X_train_{embedding_backend}.npy")
    y_path = os.path.join(artifacts_dir, f"y_train_{embedding_backend}.npy")
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found at {model_path}")
        print(f"   Please train the Random Forest model first with --embedding_backend {embedding_backend}")
        sys.exit(1)
    
    if not os.path.exists(X_path) or not os.path.exists(y_path):
        print(f"❌ Error: Training data not found in {artifacts_dir}")
        print(f"   Please train the Random Forest model first.")
        sys.exit(1)
    
    # Load model and data
    print(f"📂 Loading model from {model_path}")
    model = joblib.load(model_path)
    
    print(f"📂 Loading training embeddings from {X_path}")
    X_train = np.load(X_path)
    y_train = np.load(y_path)
    
    print(f"   Loaded {X_train.shape[0]} training samples with {X_train.shape[1]} features")
    print(f"   Class distribution: {np.bincount(y_train)}")
    
    # Use a subset for SHAP calculation (full dataset can be slow)
    # For large datasets, we sample a subset for the background distribution
    if X_train.shape[0] > 500:
        print(f"   Using 500 samples for SHAP background distribution (from {X_train.shape[0]} total)")
        sample_indices = np.random.choice(X_train.shape[0], size=500, replace=False)
        X_background = X_train[sample_indices]
    else:
        X_background = X_train
    
    # Create SHAP explainer
    print("🧮 Creating SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values
    print("🧮 Calculating SHAP values...")
    shap_values = explainer.shap_values(X_background)
    
    # For binary classification, shap_values might be a list [class_0, class_1]
    # We typically explain the positive class (class 1)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Use positive class explanations
    
    print(f"✅ SHAP values computed: shape {shap_values.shape}")
    
    # --- Plot 1: SHAP Summary Plot (Beeswarm) ---
    print("📊 Generating SHAP summary plot (beeswarm)...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values,
        X_background,
        max_display=max_display,
        show=False,
        plot_type="dot"
    )
    plt.title(
        f"SHAP Summary Plot: Random Forest ({embedding_backend})\n"
        f"Each dot represents a sample, color shows feature value (red=high, blue=low)",
        fontsize=12,
        fontweight="bold",
        pad=20
    )
    plt.tight_layout()
    summary_path = os.path.join(shap_dir, f"shap_summary_{embedding_backend}.png")
    plt.savefig(summary_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Saved to {summary_path}")
    
    # --- Plot 2: SHAP Feature Importance (Bar Plot) ---
    print("📊 Generating SHAP feature importance bar plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_background,
        max_display=max_display,
        show=False,
        plot_type="bar"
    )
    plt.title(
        f"SHAP Feature Importance: Random Forest ({embedding_backend})\n"
        f"Mean absolute SHAP value per feature dimension",
        fontsize=12,
        fontweight="bold",
        pad=20
    )
    plt.tight_layout()
    bar_path = os.path.join(shap_dir, f"shap_bar_{embedding_backend}.png")
    plt.savefig(bar_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Saved to {bar_path}")
    
    # --- Plot 3: SHAP Waterfall Plot (for a single prediction) ---
    print("📊 Generating SHAP waterfall plot (example prediction)...")
    # Pick a representative sample (e.g., first sample)
    sample_idx = 0
    
    # Create Explanation object for waterfall plot
    explanation = shap.Explanation(
        values=shap_values[sample_idx],
        base_values=explainer.expected_value if not isinstance(explainer.expected_value, list) 
                    else explainer.expected_value[1],
        data=X_background[sample_idx],
        feature_names=[f"dim_{i}" for i in range(X_background.shape[1])]
    )
    
    plt.figure(figsize=(10, 8))
    shap.waterfall_plot(explanation, max_display=max_display, show=False)
    plt.title(
        f"SHAP Waterfall Plot: Random Forest ({embedding_backend})\n"
        f"Explanation for sample #{sample_idx} (true class: {y_train[sample_idx]})",
        fontsize=12,
        fontweight="bold",
        pad=20
    )
    plt.tight_layout()
    waterfall_path = os.path.join(shap_dir, f"shap_waterfall_{embedding_backend}.png")
    plt.savefig(waterfall_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Saved to {waterfall_path}")
    
    # --- Summary Statistics ---
    print("\n📈 SHAP Analysis Summary:")
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    top_k = 10
    top_features = np.argsort(mean_abs_shap)[-top_k:][::-1]
    
    print(f"\n   Top {top_k} most important feature dimensions (by mean |SHAP|):")
    for rank, feat_idx in enumerate(top_features, 1):
        print(f"   {rank:2d}. Feature dim {feat_idx:3d}: {mean_abs_shap[feat_idx]:.6f}")
    
    print(f"\n✅ SHAP analysis complete! All plots saved to {shap_dir}/")
    print(f"   - {os.path.basename(summary_path)}")
    print(f"   - {os.path.basename(bar_path)}")
    print(f"   - {os.path.basename(waterfall_path)}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate SHAP analysis for trained Random Forest model."
    )
    
    parser.add_argument(
        "-e",
        "--embedding_backend",
        default="finbert",
        choices=["finbert", "minilm"],
        help="Embedding backend used during training. Default: finbert",
    )
    
    parser.add_argument(
        "--max_display",
        type=int,
        default=20,
        help="Maximum number of features to display in plots. Default: 20",
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_shap_analysis(
        embedding_backend=args.embedding_backend,
        max_display=args.max_display
    )






