"""
v2 Classification Pipeline Runner
Executes the same logic as E2E_ML_Classification_Model_v2_clean.ipynb
and saves all outputs + visuals.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, ConfusionMatrixDisplay
)
from imblearn.over_sampling import SMOTE
import joblib

VISUALS_PATH = 'visuals/'
DATA_PATH = 'data/raw/Census.csv'
os.makedirs(VISUALS_PATH, exist_ok=True)

print("=" * 60)
print("STEP 1: Load dataset")
print("=" * 60)
df = pd.read_csv(DATA_PATH)
print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Columns: {df.columns.tolist()}")

# ── STEP 2: EDA / class distribution ──────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Class distribution")
print("=" * 60)
income_counts = df['income'].value_counts()
print(income_counts)

fig, ax = plt.subplots(figsize=(7, 4))
colors = ['#4C72B0', '#DD8452']
income_counts.plot(kind='bar', ax=ax, color=colors, edgecolor='black')
ax.set_title('Income Class Distribution (Original Dataset)', fontsize=13)
ax.set_xlabel('Income Class')
ax.set_ylabel('Count')
ax.set_xticklabels(income_counts.index, rotation=0)
for i, v in enumerate(income_counts.values):
    ax.text(i, v + 100, str(v), ha='center', fontsize=10)
plt.tight_layout()
plt.savefig(VISUALS_PATH + 'class_distribution.png', dpi=150)
plt.close()
print("Saved: visuals/class_distribution.png")

# ── STEP 3: Preprocessing ─────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Preprocessing")
print("=" * 60)
df_clean = df.copy()
df_clean = df_clean.replace(' ?', np.nan).dropna()
print(f"Rows after dropping missing values: {len(df_clean)}")

str_cols = df_clean.select_dtypes(include='object').columns
for col in str_cols:
    df_clean[col] = df_clean[col].str.strip()

df_clean['income'] = (df_clean['income'] == '>50K').astype(int)
print(f"Target encoded: <=50K=0, >50K=1")
print(df_clean['income'].value_counts())

# ── STEP 4: Separate X and y BEFORE encoding ──────────────────
print("\n" + "=" * 60)
print("STEP 4: Separate features and target (leakage check)")
print("=" * 60)
y = df_clean['income'].copy()
X_raw = df_clean.drop(columns=['income']).copy()
print(f"Feature matrix shape: {X_raw.shape}")

X_encoded = pd.get_dummies(X_raw, drop_first=True)
print(f"After one-hot encoding: {X_encoded.shape}")

# Leakage check
target_leak_cols = [c for c in X_encoded.columns if 'income' in c.lower()]
if target_leak_cols:
    print(f"WARNING: Target-derived columns found and dropped: {target_leak_cols}")
    X_encoded = X_encoded.drop(columns=target_leak_cols)
else:
    print("CONFIRMED: No target-derived columns in feature matrix. No leakage.")
print(f"Final feature matrix shape: {X_encoded.shape}")

# ── STEP 5: Train/test split BEFORE SMOTE ─────────────────────
print("\n" + "=" * 60)
print("STEP 5: Train/test split (70/30, stratified) — BEFORE SMOTE")
print("=" * 60)
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.3, random_state=42, stratify=y
)
print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples:     {X_test.shape[0]}")
print(f"Train class distribution:\n{y_train.value_counts()}")
print(f"Test class distribution:\n{y_test.value_counts()}")

# ── STEP 6: SMOTE on training data only ───────────────────────
print("\n" + "=" * 60)
print("STEP 6: SMOTE applied to training data only")
print("=" * 60)
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"After SMOTE — training samples: {X_train_res.shape[0]}")
print(f"Class distribution after SMOTE:\n{pd.Series(y_train_res).value_counts()}")
print(f"Test set unchanged: {X_test.shape[0]} samples (original data only)")

# ── STEP 7: Scaling ───────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7: StandardScaler (fit on training only)")
print("=" * 60)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled  = scaler.transform(X_test)
print(f"Train scaled shape: {X_train_scaled.shape}")
print(f"Test scaled shape:  {X_test_scaled.shape}")

# ── STEP 8: Train and evaluate models ─────────────────────────
print("\n" + "=" * 60)
print("STEP 8: Model training and evaluation")
print("=" * 60)
models = {
    'Logistic Regression':    LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest':          RandomForestClassifier(n_estimators=100, random_state=42),
    'Support Vector Machine': CalibratedClassifierCV(LinearSVC(max_iter=2000, random_state=42))
}

results = {}
trained_models = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_scaled, y_train_res)
    y_pred = model.predict(X_test_scaled)
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    results[name] = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1 Score': f1}
    trained_models[name] = model
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")

results_df = pd.DataFrame(results).T
print("\n--- Model Comparison (original test set) ---")
print(results_df.round(4).to_string())

# ── STEP 9: Confusion matrix ──────────────────────────────────
print("\n" + "=" * 60)
print("STEP 9: Confusion matrix (Logistic Regression)")
print("=" * 60)
lr_model = trained_models['Logistic Regression']
y_pred_lr = lr_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred_lr)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['<=50K', '>50K'])
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, colorbar=False, cmap='Blues')
ax.set_title('Confusion Matrix — Logistic Regression\n(v2: SMOTE on train only, original test set)', fontsize=10)
plt.tight_layout()
plt.savefig(VISUALS_PATH + 'confusion_matrix.png', dpi=150)
plt.close()
print("Saved: visuals/confusion_matrix.png")

print("\nClassification Report — Logistic Regression:")
print(classification_report(y_test, y_pred_lr, target_names=['<=50K', '>50K']))

# ── STEP 10: Classification report summary chart ──────────────
print("\n" + "=" * 60)
print("STEP 10: Classification report summary chart")
print("=" * 60)
report = classification_report(y_test, y_pred_lr, target_names=['<=50K', '>50K'], output_dict=True)
report_df = pd.DataFrame(report).T.drop(index=['accuracy', 'macro avg', 'weighted avg'], errors='ignore')
report_df = report_df[['precision', 'recall', 'f1-score']].astype(float)

fig, ax = plt.subplots(figsize=(7, 4))
report_df.plot(kind='bar', ax=ax, edgecolor='black')
ax.set_title('Classification Report — Logistic Regression (v2)', fontsize=12)
ax.set_xlabel('Class')
ax.set_ylabel('Score')
ax.set_xticklabels(report_df.index, rotation=0)
ax.set_ylim(0, 1.1)
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig(VISUALS_PATH + 'classification_report_summary.png', dpi=150)
plt.close()
print("Saved: visuals/classification_report_summary.png")

# ── STEP 11: ROC curves ───────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 11: ROC curves")
print("=" * 60)
fig, ax = plt.subplots(figsize=(7, 5))
for name, model in trained_models.items():
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
    print(f"  {name} AUC: {roc_auc:.4f}")
ax.plot([0, 1], [0, 1], 'k--', label='Random classifier')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves — All Models (v2)')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig(VISUALS_PATH + 'roc_curve.png', dpi=150)
plt.close()
print("Saved: visuals/roc_curve.png")

# ── STEP 12: Hyperparameter tuning ────────────────────────────
print("\n" + "=" * 60)
print("STEP 12: Hyperparameter tuning (Logistic Regression)")
print("=" * 60)
param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l2'],
    'solver': ['lbfgs'],
    'max_iter': [1000]
}
grid_search = GridSearchCV(
    LogisticRegression(random_state=42),
    param_grid, cv=5, scoring='f1', verbose=0, n_jobs=-1
)
grid_search.fit(X_train_scaled, y_train_res)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV F1:      {grid_search.best_score_:.4f}")

tuned_lr = grid_search.best_estimator_
y_pred_tuned = tuned_lr.predict(X_test_scaled)
print("\nTuned Logistic Regression — Test Set:")
print(f"  Accuracy:  {accuracy_score(y_test, y_pred_tuned):.4f}")
print(f"  Precision: {precision_score(y_test, y_pred_tuned):.4f}")
print(f"  Recall:    {recall_score(y_test, y_pred_tuned):.4f}")
print(f"  F1 Score:  {f1_score(y_test, y_pred_tuned):.4f}")

# ── STEP 13: Feature importance ───────────────────────────────
print("\n" + "=" * 60)
print("STEP 13: Feature importance")
print("=" * 60)
feature_names = X_encoded.columns.tolist()
coef = np.abs(tuned_lr.coef_[0])
importance_df = pd.DataFrame({'feature': feature_names, 'importance': coef})
importance_df = importance_df.sort_values('importance', ascending=False).head(20)

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(importance_df['feature'][::-1], importance_df['importance'][::-1], color='#4C72B0')
ax.set_xlabel('Absolute Coefficient Value')
ax.set_title('Top 20 Feature Importances — Logistic Regression (v2)', fontsize=12)
plt.tight_layout()
plt.savefig(VISUALS_PATH + 'feature_importance.png', dpi=150)
plt.close()
print("Saved: visuals/feature_importance.png")
print("\nTop 10 features:")
print(importance_df.head(10).to_string(index=False))

# ── STEP 14: Save model ───────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 14: Save model")
print("=" * 60)
joblib.dump(tuned_lr, 'logistic_regression_model_v2.pkl')
print("Saved: logistic_regression_model_v2.pkl")

# ── Final summary ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("PIPELINE COMPLETE — FINAL METRICS SUMMARY")
print("=" * 60)
print(results_df.round(4).to_string())
print(f"\nTuned LR — Best C: {grid_search.best_params_['C']}")
print(f"Tuned LR — Test F1: {f1_score(y_test, y_pred_tuned):.4f}")
print(f"Tuned LR — Test Accuracy: {accuracy_score(y_test, y_pred_tuned):.4f}")

# Write metrics to a temp file for README/docs update
metrics = {
    'lr_acc':   round(results['Logistic Regression']['Accuracy'], 4),
    'lr_prec':  round(results['Logistic Regression']['Precision'], 4),
    'lr_rec':   round(results['Logistic Regression']['Recall'], 4),
    'lr_f1':    round(results['Logistic Regression']['F1 Score'], 4),
    'rf_acc':   round(results['Random Forest']['Accuracy'], 4),
    'rf_prec':  round(results['Random Forest']['Precision'], 4),
    'rf_rec':   round(results['Random Forest']['Recall'], 4),
    'rf_f1':    round(results['Random Forest']['F1 Score'], 4),
    'svm_acc':  round(results['Support Vector Machine']['Accuracy'], 4),
    'svm_prec': round(results['Support Vector Machine']['Precision'], 4),
    'svm_rec':  round(results['Support Vector Machine']['Recall'], 4),
    'svm_f1':   round(results['Support Vector Machine']['F1 Score'], 4),
    'tuned_c':  grid_search.best_params_['C'],
    'tuned_acc': round(accuracy_score(y_test, y_pred_tuned), 4),
    'tuned_f1':  round(f1_score(y_test, y_pred_tuned), 4),
    'tuned_prec': round(precision_score(y_test, y_pred_tuned), 4),
    'tuned_rec':  round(recall_score(y_test, y_pred_tuned), 4),
}
import json
with open('v2_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print("\nMetrics written to v2_metrics.json")
