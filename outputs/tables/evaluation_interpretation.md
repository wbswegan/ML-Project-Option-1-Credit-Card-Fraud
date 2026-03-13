# Evaluation Interpretation



In this imbalanced fraud detection task, model quality should be judged primarily by recall and F1-score rather than accuracy alone.

The best-performing model on the holdout test set is **baseline_logistic_regression** (precision=0.0853, recall=0.7464, F1=0.1531, ROC-AUC=0.8291).

Compared with **improved_random_forest**, it offers a stronger fraud-detection tradeoff, capturing more fraudulent transactions (higher recall) while maintaining competitive precision. This balance is important because increasing recall can raise false positives.

Overall, the preferred model should be selected based on operational tolerance for false alarms versus missed fraud cases, with priority typically given to minimizing missed fraud.