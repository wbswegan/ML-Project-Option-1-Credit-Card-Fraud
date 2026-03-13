# Validation and Leakage Prevention Notes

1. **Stratified split**
- Train/test data is created with a stratified split so both sets preserve the fraud class ratio.

2. **Holdout preprocessing safety**
- The preprocessor is fit on training data only, then applied to test data.
- Test labels and test feature statistics are never used during model fitting.

3. **Cross-validation preprocessing safety**
- During CV, preprocessing and model are wrapped in one sklearn Pipeline, so each fold fits preprocessing only on that fold's training partition.
- This avoids leakage from validation folds into preprocessing parameters.

4. **CV setting used in this run**
- Cross-validation was not requested in this run (`--run-cv` not provided).