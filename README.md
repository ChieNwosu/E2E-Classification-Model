# End-to-End Machine Learning Classification: Census Income Prediction

**Tech Stack:** Python · pandas · NumPy · scikit-learn · imbalanced-learn · Plotly · matplotlib · seaborn · Google Colab · GitHub

---

## Executive Summary

This project builds a complete end-to-end machine learning classification pipeline to predict whether an individual earns ≤$50K or >$50K per year, using U.S. Census demographic and employment data. The workflow covers every stage of the ML lifecycle — from raw data exploration and preprocessing through model training, cross-validation, hyperparameter tuning, feature selection, and model persistence.

The project was completed as a structured learning exercise following the book *Mastering ChatGPT and Google Colab for Machine Learning* by Moscato. It reflects a chapter-by-chapter progression through core ML concepts, with an emphasis on building reproducible, interpretable, and ethically aware pipelines.

**This repository contains two notebook versions:**

| Version | File | Purpose |
|---|---|---|
| v1 (original) | `E2E_ML_Classification_Model.ipynb` | Chapter-based learning artifact from the textbook. Preserved unchanged for transparency. |
| v2 (clean) | `notebooks/E2E_ML_Classification_Model_v2_clean.ipynb` | Portfolio-ready version with corrected SMOTE workflow and target leakage fix. |

---

## Business Problem

Income inequality is a measurable socioeconomic outcome with real policy implications. Predicting income level from demographic and employment features can support research in labor economics, workforce development, and social services — but it also carries significant ethical responsibility.

This project frames the task as a binary classification problem:

- **Target variable:** `income` — whether a person earns `<=50K` or `>50K` per year
- **Task type:** Supervised binary classification
- **Primary challenge:** Class imbalance, correlated demographic features, and the need for metric-aware evaluation

---

## Dataset Overview

**Source:** U.S. Census Income dataset (`Census.csv`)

| Property | Detail |
|---|---|
| Rows | ~32,561 |
| Features | 14 input features + 1 target |
| Target | `income` (<=50K / >50K) |
| Class distribution | Imbalanced (~75% <=50K, ~25% >50K) |

**Features include:**
- `age`, `education`, `education-num`
- `workclass`, `occupation`, `hours-per-week`
- `marital-status`, `relationship`
- `race`, `sex`, `native-country`
- `capital-gain`, `capital-loss`, `final-weight`

> Note: This dataset contains sensitive demographic attributes. Class imbalance and correlated features make evaluation metric selection especially important.

---

## Learning Context

This project was completed while working through *Mastering ChatGPT and Google Colab for Machine Learning* by Moscato. The notebook follows a chapter-based learning structure, where each chapter introduces a new stage of the ML workflow — from data loading and exploration through preprocessing, model training, evaluation, and refinement.

Prompts from the textbook are cited directly in the notebook cells to preserve the instructional context and show how AI-assisted coding was used as a learning tool. This is intentional: the project is designed as a learning-to-portfolio artifact, not a production system.

---

## Machine Learning Workflow

The notebook is organized into chapters that mirror the ML lifecycle:

| Chapter | Focus |
|---|---|
| Ch. 3–4 | Data loading, exploration, missing value handling, outlier analysis |
| Ch. 5 | Exploratory data analysis (EDA), visualizations, class distribution |
| Ch. 6 | Preprocessing pipeline: encoding, SMOTE, standardization |
| Ch. 7 | Model training, evaluation, cross-validation |
| Ch. 8 | Hyperparameter tuning, feature selection, model persistence |

---

## Data Preprocessing

The preprocessing pipeline applies the following steps in sequence:

1. **Label Encoding** — converts categorical columns to numeric form
2. **One-Hot Encoding** — expands categorical variables to binary columns
3. **SMOTE (Synthetic Minority Oversampling Technique)** — addresses class imbalance by generating synthetic minority-class samples, resulting in a balanced dataset of 49,440 rows (24,720 per class)
4. **StandardScaler** — standardizes numerical features to zero mean and unit variance
5. **Pipeline persistence** — the preprocessed dataset is saved as `preprocessed_dataset.csv` for reproducible model training

---

## Model Training

Three classification algorithms were trained and compared using a 70/30 train/test split:

| Model | Notes |
|---|---|
| Logistic Regression | Baseline linear classifier; tuned with GridSearchCV |
| Random Forest | Ensemble tree-based classifier |
| Support Vector Machine (SVC) | Kernel-based classifier |

**Train/test split:** 70% training (34,608 samples), 30% testing (14,832 samples)

**Hyperparameter tuning:** GridSearchCV with 5-fold cross-validation was applied to Logistic Regression. Best parameters found: `C=0.1`, `penalty=l2`, `solver=lbfgs`.

---

## Model Evaluation

> **Note on results:** The evaluation outputs below are taken directly from notebook cell outputs. The near-perfect scores on the preprocessed dataset are likely influenced by the SMOTE oversampling applied before splitting — a known data leakage risk in this pipeline design. These results should be interpreted as a learning artifact, not as production-ready performance claims. See the Limitations section for more detail.

### Cross-Validation Results (5-Fold, Preprocessed Dataset)

| Model | Mean Accuracy | Std Dev |
|---|---|---|
| Logistic Regression | 1.0000 | 0.0000 |
| Random Forest | 1.0000 | 0.0000 |
| Support Vector Machine | 0.9958 | 0.0007 |

### SVM Detailed Metrics (Cross-Validation)

| Metric | Mean | Std Dev |
|---|---|---|
| Accuracy | 0.9958 | 0.0007 |
| Precision | 0.9975 | 0.0006 |
| Recall | 0.9940 | 0.0015 |
| F1 Score | 0.9958 | 0.0007 |

### Overfitting Check (Logistic Regression, Best Model)

| Split | Accuracy |
|---|---|
| Training | 1.0 |
| Test | 1.0 |
| Verdict | Well-fit (no overfitting detected) |

> **TODO:** Verify these results by re-running the notebook. Consider re-evaluating on the original (non-SMOTE) test set to get a more realistic performance estimate.

---

## Ethical Considerations

Income prediction using Census-style demographic data requires careful consideration. The features in this dataset — including race, sex, marital status, native country, and occupation — reflect real-world social and economic patterns that are shaped by historical inequalities, systemic bias, and structural barriers.

A model trained on this data can unintentionally learn and reinforce those patterns. For example:

- Predicting lower income for individuals from certain demographic groups may reflect historical discrimination rather than individual capability
- Using such a model in hiring, lending, or benefits decisions without human oversight could cause real harm
- Class imbalance in the original dataset means the model may be less reliable for the minority (>50K) class without careful handling

This project applies SMOTE to address class imbalance and evaluates models using precision, recall, and F1-score rather than accuracy alone. However, it does not include formal fairness auditing, disparate impact analysis, or demographic parity checks.

**This project is intended for learning and portfolio purposes only. It should not be used for automated decision-making.**

---

## Limitations

- **SMOTE applied before train/test split:** Synthetic samples from the minority class may appear in both training and test sets, inflating evaluation metrics. This is a known data leakage issue in the current pipeline design.
- **Near-perfect scores:** The 1.0 accuracy results for Logistic Regression and Random Forest are likely a consequence of the above. Real-world performance on unseen Census data would be lower.
- **No formal fairness analysis:** Demographic parity, equalized odds, and other fairness metrics are not computed.
- **Google Colab environment:** The notebook was designed for Colab and uses `files.upload()` for data loading. Local execution requires path adjustments.
- **No production deployment:** The model is saved locally using `joblib` but is not deployed to any API or cloud service.
- **Feature leakage risk:** The `income_1` column (a one-hot encoded version of the target) may have been included in the feature set during training. This should be verified.

---

## What I Learned

- How to build a complete ML pipeline from raw data to a saved model
- Why preprocessing order matters — and how SMOTE placement affects evaluation integrity
- How to compare multiple classifiers using consistent, multi-metric evaluation
- Why accuracy alone is misleading for imbalanced classification problems
- How cross-validation provides more reliable performance estimates than a single train/test split
- How hyperparameter tuning with GridSearchCV improves model configuration
- How feature importance from logistic regression coefficients can guide feature selection
- Why ethical awareness is not optional when working with demographic data

---

## How to Run This Project

**Environment:** Google Colab (v1) or local Python environment (v2)

**Requirements:**
```bash
pip install -r requirements.txt
```

### Running the original notebook (v1 — learning version)

1. Open `E2E_ML_Classification_Model.ipynb` in Google Colab or Jupyter
2. Upload `Census.csv` when prompted (or update the file path for local use)
3. Run cells sequentially — each chapter builds on the previous
4. The preprocessed dataset will be saved as `preprocessed_dataset.csv`
5. The final model will be saved as `logistic_regression_model.pkl`

> Note: Some cells use `from google.colab import files` for file uploads. If running locally, replace these with `pd.read_csv("Census.csv")`.

### Running the v2 clean notebook (portfolio version)

1. Open `notebooks/E2E_ML_Classification_Model_v2_clean.ipynb` in Jupyter or VS Code
2. The dataset is read from `data/raw/Census.csv` — no upload needed
3. Run all cells sequentially
4. Charts will be saved automatically to `visuals/`
5. The tuned model will be saved as `logistic_regression_model_v2.pkl`

---

## Recommended Visuals

> The following visuals are generated within the notebook. They have not been exported to the `visuals/` folder yet.

**TODO — Export these from the notebook and save to `visuals/`:**
- [ ] Class distribution bar chart (income <=50K vs >50K)
- [ ] Feature importance bar chart (logistic regression coefficients)
- [ ] Correlation heatmap (numerical features)
- [ ] Income distribution by education level
- [ ] Income distribution by age group
- [ ] Confusion matrix (after model evaluation)
- [ ] ROC curve (if added in future iteration)

---

## Portfolio Context

This classification project is part of a broader portfolio demonstrating skills across the data and ML lifecycle:

| Project | Focus |
|---|---|
| This project | End-to-end ML classification, preprocessing pipelines, model evaluation |
| Regression ML project | Supervised regression modeling and feature engineering |
| [AWS Consumer Complaint Intelligence Dashboard](https://github.com/ChieNwosu/aws-consumer-complaint-intelligence) | Cloud analytics, AWS services, NLP-based complaint classification |

Together, these projects reflect a developing skill set in:
- Data analytics and exploratory analysis
- Supervised machine learning (classification and regression)
- Cloud-based analytics and data engineering
- Responsible AI practices and ethical awareness

---

## Next Steps

- [ ] Re-evaluate models on the original (pre-SMOTE) test set to get realistic performance estimates
- [ ] Add confusion matrix visualization and export to `visuals/`
- [ ] Add ROC-AUC curve for binary classification evaluation
- [ ] Perform fairness analysis across demographic subgroups
- [ ] Experiment with additional classifiers (e.g., XGBoost, Decision Tree)
- [ ] Refactor notebook to apply SMOTE only on training data (fix data leakage)
- [ ] Add a requirements.txt-compatible local execution path

---

## Project Structure

```
E2E-Classification-Model/
├── E2E_ML_Classification_Model.ipynb        # v1: Original chapter-based learning notebook (preserved)
├── Census.csv                               # Raw dataset (root copy for Colab compatibility)
├── requirements.txt                         # Python dependencies
├── README.md                                # This file
├── README_original.md                       # Original README (preserved)
├── data/
│   └── raw/
│       └── Census.csv                       # Raw dataset (local path for v2 notebook)
├── notebooks/
│   └── E2E_ML_Classification_Model_v2_clean.ipynb  # v2: Portfolio-ready, leakage-corrected
├── visuals/                                 # Exported charts (generated by v2 notebook)
│   ├── class_distribution.png              # TODO: run v2 notebook to generate
│   ├── confusion_matrix.png                # TODO: run v2 notebook to generate
│   ├── roc_curve.png                       # TODO: run v2 notebook to generate
│   └── feature_importance.png             # TODO: run v2 notebook to generate
└── docs/
    ├── project_overview.md
    ├── model_evaluation.md
    ├── ethical_considerations.md
    ├── limitations.md
    └── stakeholder_summary.md
```
