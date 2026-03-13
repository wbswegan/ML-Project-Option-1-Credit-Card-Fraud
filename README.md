# Credit Card Fraud Detection for Imbalanced Binary Classification

## 1. Project Title
Credit Card Fraud Detection for Imbalanced Binary Classification

## 2. Problem Statement
Credit card fraud detection is a high-impact binary classification problem with severe class imbalance. The objective of this project is to build a reproducible machine learning pipeline that identifies fraudulent transactions while minimizing missed fraud cases. We compare a simple baseline model with a stronger tree-based model using fraud-relevant metrics.

## 3. Dataset Source
- **Dataset files used in this project**: `fraudTrain.csv`, `fraudTest.csv`
- **Source**: Locally provided by the project team (already downloaded)
- **Expected local paths**:
  - `data/raw/fraudTrain.csv`
  - `data/raw/fraudTest.csv`

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
3. Ensure the local dataset files are placed in:
```text
data/raw/fraudTrain.csv
data/raw/fraudTest.csv
```

## 6. How to Run EDA, Training, and Evaluation
Run in the following order:
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
- **Baseline**: Logistic Regression (`class_weight='balanced'`)
- **Improved**: Random Forest (`class_weight='balanced_subsample'`)

Both models are evaluated on the same stratified holdout split to ensure fair comparison.

## 8. Key Results
Primary result files:
- `outputs/tables/evaluation_metrics_summary.csv`
- `outputs/tables/model_comparison.csv`
- `outputs/figures/model_roc_comparison.png`
- `outputs/figures/*_confusion_matrix.png`
- `outputs/tables/evaluation_interpretation.md`

Report guidance:
- Use **recall** and **F1-score** as primary criteria due to class imbalance.
- Treat **accuracy** as secondary, since high accuracy can occur even when fraud detection is poor.
- Use confusion matrices and ROC comparison to explain the precision-recall tradeoff in operational terms.

## 9. Limitations
- Dataset is highly imbalanced, which can still bias learning despite class-weighted models.
- Current setup uses fixed model configurations; no extensive hyperparameter search is included.
- Performance is evaluated on one dataset setting; external generalization is not guaranteed.
- Cost-sensitive thresholds are not fully optimized for a specific business workflow.

## 10. Team Contribution Placeholder
Fill this section before submission:

| Team Member | Responsibilities | Contribution Summary |
|---|---|---|
| Member A | EDA, data quality checks | _Add details_ |
| Member B | Preprocessing and leakage prevention | _Add details_ |
| Member C | Model training and evaluation | _Add details_ |
| Member D | Report writing and presentation | _Add details_ |
