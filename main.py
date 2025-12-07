import argparse
import gc

import torch

from cnn.cnn import CNN
from data_integ import evaluate, load_data
from lstm.lstm import LSTM
from randomforest.randomforest import RandomForest
from svm.svm import SVM
# from svm.svm_rbf import SVM_RBF
# from svm.svm_fuzzy import SVM_Fuzzy

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15


def main(args):
    data_path = args.data
    task_type = args.task_type
    embedding_backend = args.embedding_backend

    print(f"Loading data from {data_path} for task: {task_type}")
    dataset = load_data(data_path, task_type=task_type)
    print(f"Loaded {len(dataset)} samples from {data_path}.")

    models = []
    pipeline = args.pipeline
    if pipeline == "svm" or pipeline == "all":
        models.append(("svm", SVM))
    # if pipeline == "svm_rbf" or pipeline == "all":
    #     models.append(("svm_rbf", SVM_RBF))
    # if pipeline == "svm_fuzzy" or pipeline == "all":
    #     models.append(("svm_fuzzy", SVM_Fuzzy))
    if pipeline == "cnn" or pipeline == "all":
        models.append(("cnn", CNN))
    if pipeline == "lstm" or pipeline == "all":
        models.append(("lstm", LSTM))
    if pipeline == "randomforest" or pipeline == "all":
        models.append(("randomforest", RandomForest))
    if not len(models):
        raise ValueError(f"Unknown pipeline type: {args.pipeline}")

    n = len(dataset)

    train_n = int(n * TRAIN_RATIO)
    val_n = int(n * VAL_RATIO)

    if train_n + val_n > n:
        raise ValueError("TRAIN_RATIO + VAL_RATIO must be <= 1.0")

    train_df = dataset.iloc[:train_n].copy()
    val_df = dataset.iloc[train_n : train_n + val_n].copy()
    test_df = dataset.iloc[train_n + val_n :].copy()

    for name, modelClass in models:
        for run in range(args.runs):
            model = modelClass(embedding_backend=embedding_backend)
            print(f"Starting training run {run + 1} for {name} with {embedding_backend} embeddings.")
            model.train(train_df, val_df)

            print(f"Starting prediction run {run + 1} for {name}.")
            results = model.predict(test_df)
            evaluate(f"{name}-{embedding_backend}-{task_type}-run{run + 1}", results, task_type=task_type)

            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(description="Run ML pipelines.")

    parser.add_argument(
        "-p",
        "--pipeline",
        required=True,
        choices=["svm", "cnn", "lstm", "randomforest", "all"],  # "svm_rbf", "svm_fuzzy" commented out
        help="Select which pipeline to run.",
    )

    parser.add_argument(
        "-d",
        "--data",
        required=True,
        help="Path to the training data.",
    )

    parser.add_argument(
        "-r",
        "--runs",
        required=True,
        type=int,
        help="Number of runs to execute for each pipeline.",
    )
    
    parser.add_argument(
        "-t",
        "--task_type",
        default="1d",
        choices=["1d", "5d"],
        help="Prediction task: '1d' (1-day, Close>=Open) or '5d' (5-day forward return>=0). Default: 1d",
    )
    
    parser.add_argument(
        "-e",
        "--embedding_backend",
        default="finbert",
        choices=["finbert", "minilm"],
        help="Embedding model: 'finbert' (finance-specific) or 'minilm' (general-purpose). Default: finbert",
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
