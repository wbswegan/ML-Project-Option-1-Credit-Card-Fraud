# Credit Card Fraud Detection for Imbalanced Binary Classification

## 1. Project Title
Credit Card Fraud Detection for Imbalanced Binary Classification

## 2. Problem Statement
Credit card fraud detection is a high-impact binary classification problem with severe class imbalance. The objective of this project is to build a reproducible machine learning pipeline that identifies fraudulent transactions while minimizing missed fraud cases. A simple baseline model is compared with a stronger tree-based model using fraud-relevant evaluation metrics.

## 3. Dataset Source
- Dataset used: `fraudTrain.csv`, `fraudTest.csv`
- Kaggle link: https://www.kaggle.com/datasets/kartik2112/fraud-detection
- Local paths used by this project:
  - `data/raw/fraudTrain.csv`
  - `data/raw/fraudTest.csv`

Note: This repository does not auto-download data. Please place the dataset files manually in `data/raw/`.

## 4. Repository Structure
```text
.
|-- data/
|   |-- raw/                     # Raw dataset files (fraudTrain.csv, fraudTest.csv)
|   `-- processed/               # Preprocessed arrays and split metadata
|-- notebooks/                   # Optional notebooks
|-- src/
|   |-- config.py                # Central paths and constants
|   |-- utils.py                 # Shared helper functions
|   |-- eda.py                   # EDA workflow and findings export
|   |-- data_preprocessing.py    # Stratified split + preprocessing pipeline
|   |-- train.py                 # Baseline and improved model training
|   `-- evaluate.py              # Holdout evaluation + optional stratified CV
|-- outputs/
|   |-- figures/                 # Plots (EDA, confusion matrices, ROC, importance)
|   |-- tables/                  # Metrics, comparisons, summaries, notes
|   `-- models/                  # Serialized trained models
|-- requirements.txt
|-- README.md
`-- .gitignore
```

## 5. Installation Instructions
1. Create and activate a Python environment (Python 3.10+ recommended).
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Ensure dataset files are placed at:
```text
data/raw/fraudTrain.csv
data/raw/fraudTest.csv
```

## 6. How to Run EDA, Training, and Evaluation
Run in this order:
```bash
python src/eda.py
python src/data_preprocessing.py
python src/train.py
python src/evaluate.py
```

Optional stronger validation (stratified k-fold CV):
```bash
python src/evaluate.py --run-cv --cv-folds 5
```

Optional custom dataset path:
```bash
python src/eda.py --data-path path/to/data/raw
python src/data_preprocessing.py --data-path path/to/data/raw
python src/train.py --data-path path/to/data/raw
python src/evaluate.py --data-path path/to/data/raw
```

## 7. Models Used
- Baseline: Logistic Regression (`class_weight='balanced'`)
- Improved: Random Forest (`class_weight='balanced_subsample'`)

Both models are evaluated on the same stratified holdout split for fair comparison.

## 8. Key Results
Primary result files:
- `outputs/tables/evaluation_metrics_summary.csv`
- `outputs/tables/model_comparison.csv`
- `outputs/figures/model_roc_comparison.png`
- `outputs/figures/*_confusion_matrix.png`
- `outputs/tables/evaluation_interpretation.md`

Interpretation focus:
- Recall and F1-score are prioritized under class imbalance.
- Accuracy is secondary because it can be misleading in fraud detection.
- Confusion matrices and ROC curves are used to explain the precision-recall tradeoff.

## 9. Limitations
- The dataset is highly imbalanced, which can still bias learning despite imbalance-aware settings.
- The current setup uses fixed model configurations rather than extensive hyperparameter search.
- Performance is evaluated on one dataset setting; external generalization is not guaranteed.
- Thresholds are not fully tuned for a specific business cost function.
