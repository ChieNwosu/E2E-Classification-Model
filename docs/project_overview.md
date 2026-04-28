# Project Overview

## End-to-End Machine Learning Classification: Census Income Prediction

### Summary

This project implements a complete supervised machine learning pipeline to predict whether an individual earns ≤$50K or >$50K per year, using U.S. Census demographic and employment data.

It was completed as a structured learning exercise following *Mastering ChatGPT and Google Colab for Machine Learning* by Moscato. The notebook is organized chapter-by-chapter, with each chapter introducing a new stage of the ML workflow.

### Objective

Build a binary classification model that predicts income level from demographic and employment features, while demonstrating responsible ML practices including multi-metric evaluation, class imbalance handling, and ethical awareness.

### Dataset

- **File:** `Census.csv`
- **Rows:** ~32,561
- **Features:** 14 input features + 1 target (`income`)
- **Target classes:** `<=50K` (majority) and `>50K` (minority)
- **Class imbalance:** ~75% / ~25% split in the original dataset

### Workflow Summary

| Stage | Description |
|---|---|
| Data Loading | Loaded via Google Colab file upload; explored structure and types |
| EDA | Analyzed distributions, correlations, and class balance |
| Preprocessing | Label encoding, one-hot encoding, SMOTE, standardization |
| Model Training | Logistic Regression, Random Forest, SVM |
| Evaluation | Accuracy, precision, recall, F1-score, cross-validation |
| Tuning | GridSearchCV hyperparameter optimization |
| Feature Selection | Logistic regression coefficient-based importance ranking |
| Model Persistence | Saved with `joblib` as `logistic_regression_model.pkl` |

### Tools and Libraries

| Tool | Purpose |
|---|---|
| Python | Core language |
| pandas | Data manipulation |
| NumPy | Numerical operations |
| scikit-learn | ML models, preprocessing, evaluation |
| imbalanced-learn | SMOTE for class imbalance |
| matplotlib / seaborn | Static visualizations |
| Plotly | Interactive visualizations |
| joblib | Model serialization |
| Google Colab | Development environment |

### Learning Context

This project follows a textbook-guided workflow. Prompts from the book are cited in notebook cells to show how AI-assisted coding was used as a learning tool. The chapter structure is preserved intentionally to reflect the learning progression.
