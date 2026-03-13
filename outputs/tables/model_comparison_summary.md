# Model Comparison Summary

Fraud detection is class-imbalanced, so recall and F1 are prioritized over accuracy.

Best holdout model: **baseline_logistic_regression** (Recall=0.7464, F1=0.1531, ROC-AUC=0.8291).
Compared to **improved_random_forest**, the selected model is preferred because it captures more fraud cases (higher recall) while keeping a better precision-recall balance (higher F1).

Recommendation: use the selected model for fraud screening when missing fraud is costlier than reviewing additional flagged transactions.