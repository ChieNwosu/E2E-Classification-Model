# Model Evaluation

## Two Versions of This Project

This repository contains two notebook versions with different pipeline designs. Their evaluation results are not directly comparable because the underlying methodology differs.

| | v1 (Original) | v2 (Clean) |
|---|---|---|
| File | `E2E_ML_Classification_Model.ipynb` | `notebooks/E2E_ML_Classification_Model_v2_clean.ipynb` |
| SMOTE placement | Before train/test split | After split, training data only |
| Test set | May contain synthetic samples | Original data only |
| Target leakage | `income_1` potentially in X | Explicitly removed before encoding |
| Scaler fit | Not explicitly controlled | Fit on training data only |
| Evaluation reliability | Inflated | Realistic |

---

## Why SMOTE Before Split Causes Data Leakage

SMOTE generates synthetic samples by interpolating between existing minority-class examples. When applied to the full dataset before splitting:

1. Synthetic samples derived from real training examples can end up in the test set
2. The model is evaluated on data that is statistically similar to what it was trained on
3. This inflates all evaluation metrics — accuracy, precision, recall, and F1-score

**The correct approach:** Split first, then apply SMOTE only to the training set. The test set must contain only original, real data to provide an honest estimate of how the model would perform on unseen examples.

---

## Why `income_1` Must Not Appear in X

The original preprocessing pipeline applied one-hot encoding to all categorical columns, including the target variable `income`. This created an `income_1` column — a direct numeric encoding of the answer the model is trying to predict.

Including `income_1` in the feature matrix `X` is **direct target leakage**: the model learns to predict income by looking at a transformed version of income itself. This produces artificially perfect or near-perfect scores that have no real-world meaning.

**The correct approach:** Separate the target variable from features before any encoding. The v2 notebook does this explicitly and includes a check that verifies no target-derived columns are present in X.

---

## v1 Results (Original Notebook — Inflated Due to Data Leakage)

> These results are taken directly from notebook cell outputs. They are preserved for transparency but should not be interpreted as real-world performance estimates.

### Single Train/Test Split

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Logistic Regression | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Random Forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Support Vector Machine | 0.9949 | 0.9976 | 0.9923 | 0.9949 |

### 5-Fold Cross-Validation

| Model | Mean Accuracy | Std Dev |
|---|---|---|
| Logistic Regression | 1.0000 | 0.0000 |
| Random Forest | 1.0000 | 0.0000 |
| Support Vector Machine | 0.9958 | 0.0007 |

### SVM Detailed Cross-Validation Metrics

| Metric | Mean | Std Dev |
|---|---|---|
| Accuracy | 0.9958 | 0.0007 |
| Precision | 0.9975 | 0.0006 |
| Recall | 0.9940 | 0.0015 |
| F1 Score | 0.9958 | 0.0007 |

### Hyperparameter Tuning (Logistic Regression)

Best parameters: `C=0.1`, `penalty=l2`, `solver=lbfgs`  
Best CV accuracy: 1.0 | Test accuracy: 1.0

---

## v2 Results (Clean Notebook — Corrected Pipeline)

> **TODO:** Run all cells in `notebooks/E2E_ML_Classification_Model_v2_clean.ipynb` and update this section with the actual output metrics.

Expected realistic performance range based on published benchmarks for this dataset: **~82–87% accuracy**.

### Placeholder Table (update after running v2 notebook)

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Logistic Regression | TODO | TODO | TODO | TODO |
| Random Forest | TODO | TODO | TODO | TODO |
| Support Vector Machine | TODO | TODO | TODO | TODO |

### Tuned Logistic Regression (v2)

| Metric | Value |
|---|---|
| Best C | TODO |
| Test Accuracy | TODO |
| Test F1 Score | TODO |

---

## Visuals

> **TODO:** Run the v2 notebook to generate and save these charts to `visuals/`.

- `visuals/class_distribution.png` — Income class distribution in original dataset
- `visuals/confusion_matrix.png` — Confusion matrix for best model on original test set
- `visuals/roc_curve.png` — ROC curves for all three models
- `visuals/feature_importance.png` — Top 20 features by logistic regression coefficient magnitude
