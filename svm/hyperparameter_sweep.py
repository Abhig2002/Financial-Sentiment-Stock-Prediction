"""
Hyperparameter sweep for RBF SVM.

Example usage:
    python -m svm.hyperparameter_sweep --data ./data --output ./results
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
from svm.svm_rbf import SVM_RBF


def run_sweep(data_path: str, output_dir: str):
    """
    Run hyperparameter sweep for RBF SVM.
    
    Args:
        data_path: Path to data directory
        output_dir: Path to save results
    """
    from data_integ import load_data
    
    # Load data
    print(f"Loading data from {data_path}...")
    dataset = load_data(data_path)
    
    # Split data (70% train, 15% val, 15% test)
    n = len(dataset)
    train_n = int(n * 0.7)
    val_n = int(n * 0.15)
    
    train_df = dataset.iloc[:train_n].copy()
    val_df = dataset.iloc[train_n : train_n + val_n].copy()
    test_df = dataset.iloc[train_n + val_n :].copy()
    
    print(f"Dataset split: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")
    
    # Hyperparameter grid
    C_values = [0.1, 1.0, 10.0, 100.0]
    gamma_values = ["scale", "auto", 0.001, 0.01, 0.1]
    
    results = []
    
    total_experiments = len(C_values) * len(gamma_values)
    current = 0
    
    print(f"\nRunning {total_experiments} experiments...\n")
    
    for C in C_values:
        for gamma in gamma_values:
            current += 1
            print(f"[{current}/{total_experiments}] Training SVM_RBF(C={C}, gamma={gamma})...")
            
            try:
                # Train model
                model = SVM_RBF(C=C, gamma=gamma)
                model.train(train_df, val_df)
                
                # Evaluate on validation set
                val_preds = model.predict(val_df)
                val_precision = precision_score(val_preds["Truth"], val_preds["Prediction"], zero_division=0)
                val_recall = recall_score(val_preds["Truth"], val_preds["Prediction"], zero_division=0)
                val_f1 = f1_score(val_preds["Truth"], val_preds["Prediction"], zero_division=0)
                
                # Evaluate on test set
                test_preds = model.predict(test_df)
                test_precision = precision_score(test_preds["Truth"], test_preds["Prediction"], zero_division=0)
                test_recall = recall_score(test_preds["Truth"], test_preds["Prediction"], zero_division=0)
                test_f1 = f1_score(test_preds["Truth"], test_preds["Prediction"], zero_division=0)
                
                results.append({
                    "C": C,
                    "gamma": gamma,
                    "val_precision": val_precision,
                    "val_recall": val_recall,
                    "val_f1": val_f1,
                    "test_precision": test_precision,
                    "test_recall": test_recall,
                    "test_f1": test_f1,
                })
                
                print(f"  Val F1: {val_f1:.4f}, Test F1: {test_f1:.4f}\n")
                
            except Exception as e:
                print(f"  ERROR: {e}\n")
                results.append({
                    "C": C,
                    "gamma": gamma,
                    "val_precision": 0,
                    "val_recall": 0,
                    "val_f1": 0,
                    "test_precision": 0,
                    "test_recall": 0,
                    "test_f1": 0,
                    "error": str(e)
                })
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results_df = pd.DataFrame(results)
    output_path = os.path.join(output_dir, "rbf_svm_sweep_results.csv")
    results_df.to_csv(output_path, index=False)
    
    print(f"\n✅ Saved results to {output_path}")
    
    # Print best configuration
    best_idx = results_df["val_f1"].idxmax()
    best_config = results_df.iloc[best_idx]
    
    print("\n" + "="*60)
    print("BEST CONFIGURATION (based on validation F1):")
    print("="*60)
    print(f"C: {best_config['C']}")
    print(f"gamma: {best_config['gamma']}")
    print(f"Validation F1: {best_config['val_f1']:.4f}")
    print(f"Test F1: {best_config['test_f1']:.4f}")
    print("="*60)
    
    # Generate visualization
    _plot_sweep_results(results_df, output_dir)
    
    return results_df


def _plot_sweep_results(results_df: pd.DataFrame, output_dir: str):
    """
    Create visualization of hyperparameter sweep results.
    
    Args:
        results_df: DataFrame with sweep results
        output_dir: Directory to save plots
    """
    # Create figure with 2 subplots (validation and test F1 scores)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("RBF SVM Hyperparameter Sweep Results", fontsize=16, fontweight="bold")
    
    # Separate numeric and string gamma values
    numeric_gamma = results_df[results_df['gamma'].apply(lambda x: isinstance(x, (int, float)))]
    string_gamma = results_df[results_df['gamma'].apply(lambda x: isinstance(x, str))]
    
    # Plot 1: Validation F1 scores
    ax = axes[0]
    
    # Plot numeric gamma values
    for gamma in numeric_gamma['gamma'].unique():
        subset = numeric_gamma[numeric_gamma['gamma'] == gamma]
        ax.plot(subset['C'], subset['val_f1'], marker='o', linewidth=2, label=f'gamma={gamma}')
    
    # Plot string gamma values
    for gamma in string_gamma['gamma'].unique():
        subset = string_gamma[string_gamma['gamma'] == gamma]
        ax.plot(subset['C'], subset['val_f1'], marker='s', linewidth=2, linestyle='--', label=f'gamma={gamma}')
    
    ax.set_xlabel('C (Regularization)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Validation F1 Score', fontsize=12, fontweight='bold')
    ax.set_title('Validation Performance', fontsize=13, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    ax.set_ylim([0, 1.05])
    
    # Plot 2: Test F1 scores
    ax = axes[1]
    
    # Plot numeric gamma values
    for gamma in numeric_gamma['gamma'].unique():
        subset = numeric_gamma[numeric_gamma['gamma'] == gamma]
        ax.plot(subset['C'], subset['test_f1'], marker='o', linewidth=2, label=f'gamma={gamma}')
    
    # Plot string gamma values
    for gamma in string_gamma['gamma'].unique():
        subset = string_gamma[string_gamma['gamma'] == gamma]
        ax.plot(subset['C'], subset['test_f1'], marker='s', linewidth=2, linestyle='--', label=f'gamma={gamma}')
    
    ax.set_xlabel('C (Regularization)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Test F1 Score', fontsize=12, fontweight='bold')
    ax.set_title('Test Performance', fontsize=13, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    
    # Save figure
    plot_path = os.path.join(output_dir, "hyperparameter_sweep_visualization.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Saved sweep visualization to {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="RBF SVM Hyperparameter Sweep")
    
    parser.add_argument(
        "-d", "--data",
        required=True,
        help="Path to the training data directory"
    )
    
    parser.add_argument(
        "-o", "--output",
        default="./output/svm_rbf_sweep",
        help="Directory to save results (default: ./output/svm_rbf_sweep)"
    )
    
    args = parser.parse_args()
    run_sweep(args.data, args.output)


if __name__ == "__main__":
    main()

