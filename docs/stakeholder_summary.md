# Stakeholder Summary

## Project: End-to-End ML Classification — Census Income Prediction

**Prepared by:** Chiemela Joseph Nwosu
**Purpose:** Non-technical summary for portfolio reviewers, hiring managers, and collaborators

---

## What This Project Does

This project uses machine learning to predict whether a person earns more or less than $50,000 per year, based on demographic and employment information from U.S. Census data.

It demonstrates a complete data science workflow — from loading and cleaning raw data, through building and evaluating predictive models, to saving a trained model for future use.

---

## Why It Matters

Understanding income patterns has real-world applications in labor economics, workforce development, and social policy research. However, predicting income from demographic data also carries ethical risks, which this project addresses explicitly.

---

## What Was Built

A machine learning pipeline that:

1. **Loads and explores** raw Census data (~32,000 records)
2. **Cleans and preprocesses** the data — handling categorical variables, addressing class imbalance, and standardizing numerical features
3. **Trains three classification models** — Logistic Regression, Random Forest, and Support Vector Machine
4. **Evaluates model performance** using multiple metrics (accuracy, precision, recall, F1-score) and cross-validation
5. **Tunes the best model** using automated hyperparameter search
6. **Identifies the most important features** for predicting income
7. **Saves the trained model** for future use

---

## Key Results

> Note: The v1 evaluation results come from a preprocessed dataset where class balancing was applied before splitting. This inflates the metrics. See the Limitations document for context.

### v1 Results (Original Notebook — Inflated)

| Model | Accuracy (CV) |
|---|---|
| Logistic Regression | ~100% (on preprocessed data — inflated) |
| Random Forest | ~100% (on preprocessed data — inflated) |
| Support Vector Machine | ~99.6% (on preprocessed data — inflated) |

### v2 Results (Clean Notebook — Realistic)

Evaluated on the original (non-synthetic) test set of 9,049 samples:

| Model | Accuracy | F1 Score |
|---|---|---|
| Logistic Regression | 84.3% | 0.672 |
| Random Forest | 84.6% | 0.683 |
| Support Vector Machine (Linear) | 84.3% | 0.628 |

All three models achieved ROC-AUC scores between 0.89 and 0.90, indicating strong ability to distinguish between income classes. Random Forest performed best overall on F1 and AUC.

---

## Two Versions of This Project

| Version | Purpose |
|---|---|
| `E2E_ML_Classification_Model.ipynb` | Original chapter-based learning notebook. Preserved as a transparent record of the learning process. |
| `notebooks/E2E_ML_Classification_Model_v2_clean.ipynb` | Portfolio-ready version. Corrects SMOTE placement and target leakage. Produces reliable evaluation metrics. |

The v2 notebook improves reliability by:
- Applying SMOTE only to training data (not the test set)
- Separating the target variable before encoding to prevent `income_1` leakage
- Fitting the scaler on training data only
- Saving evaluation charts to `visuals/`

---

## Ethical Awareness

This project explicitly acknowledges that:

- Census data contains sensitive demographic attributes (race, sex, national origin)
- Machine learning models can unintentionally reinforce historical inequalities
- Income prediction models should never be used for automated decision-making without human oversight and formal fairness review

This project is for **educational and portfolio purposes only**.

---

## Learning Context

This project was completed while working through *Mastering ChatGPT and Google Colab for Machine Learning* by Moscato. It reflects a structured, chapter-by-chapter learning progression through core ML concepts.

---

## Skills Demonstrated

- Data loading, cleaning, and exploratory analysis
- Feature engineering and preprocessing pipelines
- Supervised classification modeling
- Model evaluation with multiple metrics
- Cross-validation and hyperparameter tuning
- Feature importance analysis
- Ethical awareness in ML
- Technical documentation

---

## How This Fits the Broader Portfolio

| Project | What It Shows |
|---|---|
| This project | End-to-end ML classification, preprocessing, evaluation |
| Regression ML project | Supervised regression modeling |
| AWS Consumer Complaint Intelligence Dashboard | Cloud analytics, AWS, NLP |

Together, these projects demonstrate a developing skill set in data analytics, machine learning, cloud computing, and responsible AI.
