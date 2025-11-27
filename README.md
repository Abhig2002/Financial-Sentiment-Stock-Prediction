# Financial Sentiment Analysis Project
## About
Within the project we provide a pipeline to run financial sentiment analysis on five different machine learning methods: convolutional neural network, long short-term memory, random forest classifier, support vector machine (linear), and support vector machine (RBF kernel).

## Dataset
We use data from the [FNSPID dataset](https://github.com/Zdong104/FNSPID_Financial_News_Dataset). We provide processed datasets used during our experimentation for the [full dataset](https://drive.google.com/file/d/1UK-OwzI7j0ITMmF1IDKxxZPrneJP9x3m/view?usp=sharing) and a [fortune 500 subset](https://drive.google.com/file/d/1tBKFjc_ilOJ3La_Kd9--UURZBTTvqfO0/view?usp=share_link).

## Run
1. Download dependencies with `pip3 install -r requirements.txt`.
2. Ensure dataset is downloaded from above section.
3. Run `python3 main.py -r 2 -d ./data_utils/fortune_500 -p svm`
	We elaborate on command line arguments below.

### Example Commands
```bash
# Run linear SVM
python3 main.py -r 2 -d ./data -p svm

# Run RBF SVM (non-linear)
python3 main.py -r 2 -d ./data -p svm_rbf

# Run all models
python3 main.py -r 1 -d ./data -p all

# Run hyperparameter sweep for RBF SVM
python3 -m svm.hyperparameter_sweep -d ./data -o ./output/svm_rbf_sweep
```

## Command Line Arguments

### main.py
`-r` or `--runs`: An integer representing amount of repeats to test the data on a pipeline.

`-d` or `--data`: The path to the data folder.

`-p` or `--pipeline`: A string representing which model to use. Choices: `["svm", "svm_rbf", "cnn", "lstm", "randomforest", "all"]`

### RBF SVM Hyperparameter Sweep
`-d` or `--data`: The path to the data folder.

`-o` or `--output`: Directory to save sweep results (default: `./output/svm_rbf_sweep`)

## Models

### Support Vector Machines
- **Linear SVM** (`svm`): Fast linear classifier using LinearSVC
- **RBF SVM** (`svm_rbf`): Non-linear classifier with RBF kernel
  - Hyperparameters: C (regularization), gamma (kernel coefficient)
  - Captures non-linear decision boundaries
  - Slower but potentially more accurate

### Deep Learning
- **CNN**: 1D Convolutional Neural Network with multiple kernel sizes
- **LSTM**: Bidirectional LSTM for sequential processing

### Ensemble Methods
- **Random Forest**: 100 decision trees with validation monitoring

## Output Structure

Each model generates:
- `output/{model}-run{n}/metrics.csv` - Precision, Recall, F1 scores
- `output/{model}-run{n}/confusion_matrix.png` - Confusion matrix visualization
- `output/{model}/training_history.png` - Training graphs (CNN, LSTM, RandomForest only)
