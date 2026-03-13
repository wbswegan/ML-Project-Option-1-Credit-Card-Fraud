# EDA Findings (Fraud Detection)

1. **Strong class imbalance**
- Non-fraud: 99.4790% | Fraud: 0.5210% of all transactions.

2. **Data quality checks**
- Duplicate rows: 0
- Total missing cells: 0

3. **Amount behavior**
- Median amount (non-fraud): 47.24 | Median amount (fraud): 390.00

4. **Most class-related numeric features**
- Top correlated features with fraud label: amt, unix_time, merch_lat, lat

5. **Time pattern**
- Approximate fraud activity peak at hour: 23

These findings support using imbalance-aware metrics such as recall and F1 in modeling.