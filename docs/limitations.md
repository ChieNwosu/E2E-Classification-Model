# Limitations

## Overview

This document describes the known limitations of the current project. These are documented transparently to support honest evaluation of the work and to guide future improvements.

**Note on v2:** A cleaned portfolio version of this project (`notebooks/E2E_ML_Classification_Model_v2_clean.ipynb`) was created to address limitations 1, 2, and 3 below. The original notebook is preserved unchanged as a learning artifact. See the table at the bottom for a status summary.

---

## 1. Data Leakage: SMOTE Applied Before Train/Test Split

**Issue:** SMOTE (Synthetic Minority Oversampling Technique) was applied to the full dataset before splitting into training and test sets. This means synthetic samples generated from the minority class may appear in both the training and test sets.

**Impact:** Evaluation metrics (accuracy, precision, recall, F1-score) are likely inflated. The near-perfect scores (1.0) for Logistic Regression and Random Forest are almost certainly a consequence of this issue rather than genuine model performance.

**Recommended fix:** Apply SMOTE only to the training set, after the train/test split. The test set should always contain only original (non-synthetic) samples.

---

## 2. Near-Perfect Evaluation Scores

**Issue:** Logistic Regression and Random Forest both achieved 1.0 accuracy, precision, recall, and F1-score across all cross-validation folds.

**Impact:** These results are not credible as real-world performance estimates. They reflect the data leakage issue described above and should not be cited as evidence of model quality.

**Recommended action:** Re-run evaluation after fixing the SMOTE placement. Expect realistic accuracy in the range of 80–87% based on published benchmarks for this dataset.

---

## 3. Potential Feature Leakage: `income_1` Column

**Issue:** The preprocessing pipeline applies one-hot encoding to all categorical columns, including the target variable `income`. This creates an `income_1` column that is a direct encoding of the target. If this column was included in the feature set during training, it would cause severe data leakage.

**Recommended action:** Verify that `income_1` was dropped from the feature matrix before model training. Review the preprocessing pipeline in Chapter 6 of the notebook.

---

## 4. Google Colab Dependency

**Issue:** The notebook uses `from google.colab import files` for data upload and `files.upload()` for loading the dataset. This is specific to the Google Colab environment.

**Impact:** The notebook cannot be run locally without modifying the data loading cells.

**Recommended fix:** Replace Colab-specific upload cells with `pd.read_csv("Census.csv")` for local execution. Add a note at the top of the notebook explaining this.

---

## 5. No Formal Fairness Analysis

**Issue:** The project does not evaluate model performance across demographic subgroups (e.g., by race, sex, or native country).

**Impact:** It is unknown whether the model performs equally well for all groups. Given the nature of the dataset, disparate performance is a real risk.

**Recommended action:** Use tools like IBM AI Fairness 360 or Fairlearn to compute fairness metrics such as demographic parity, equalized odds, and disparate impact.

---

## 6. No Production Deployment

**Issue:** The model is saved locally using `joblib` but is not deployed to any API, web service, or cloud platform.

**Impact:** The model cannot be used for inference outside of the notebook environment.

**Note:** This is appropriate for a learning project. Production deployment would require additional work including input validation, API design, monitoring, and security review.

---

## 7. Historical Data Limitations

**Issue:** The Census dataset reflects income distributions from a specific historical period. The patterns in the data may not reflect current labor market conditions.

**Impact:** A model trained on this data may not generalize well to current populations.

---

## Summary Table

| Limitation | Severity | Status in v1 | Status in v2 |
|---|---|---|---|
| SMOTE before split (data leakage) | High | Known issue | Fixed — SMOTE on train only |
| Near-perfect scores | High | Consequence of above | Resolved by fix |
| Potential `income_1` feature leakage | High | Needs verification | Fixed — target separated before encoding |
| Google Colab dependency | Medium | Present | Removed — reads from `data/raw/Census.csv` |
| No fairness analysis | Medium | Out of scope | Still out of scope |
| No production deployment | Low | Intentional | Intentional |
| Historical data | Low | Inherent to dataset | Inherent to dataset |
