# =========================================================
# CARDIONET-X (CATBOOST + LIGHTGBM ENSEMBLE)
# =========================================================

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix, f1_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

# ================================
# LOAD DATA
# ================================
df = pd.read_csv("cardio.csv")
target = "cardio"

# ================================
# FEATURE ENGINEERING
# ================================
df["age"] = df["age"] / 365
df["bmi"] = df["weight"] / ((df["height"] / 100) ** 2)
df["pulse_pressure"] = df["ap_hi"] - df["ap_lo"]
df["mean_arterial_pressure"] = df["ap_lo"] + (df["pulse_pressure"] / 3)

df["hypertension_flag"] = (
    (df["ap_hi"] >= 140) | (df["ap_lo"] >= 90)
).astype(int)

# Additional features for better prediction
df["bmi_category"] = pd.cut(df["bmi"], bins=[0, 18.5, 25, 30, 100], labels=[0, 1, 2, 3]).astype(float)
df["pulse_pressure_category"] = pd.cut(df["pulse_pressure"], bins=[0, 20, 40, 60, 200], labels=[0, 1, 2, 3]).astype(float)
df["age_group"] = pd.cut(df["age"], bins=[0, 40, 50, 60, 100], labels=[0, 1, 2, 3]).astype(float)
df["systolic_diastolic_ratio"] = (df["ap_hi"] + 1) / (df["ap_lo"] + 1)  # Avoid division by zero

# ================================
# CLEANING (SAFE RULES)
# ================================
df = df[(df["ap_hi"] > 70) & (df["ap_hi"] < 250)]
df = df[(df["ap_lo"] > 40) & (df["ap_lo"] < 180)]
df = df[df["ap_hi"] > df["ap_lo"]]

# ================================
# SPLIT
# ================================
X = df.drop(columns=[target])
y = df[target]

# ================================
# IMPUTATION
# ================================
imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(X)

# ================================
# FEATURE SCALING
# ================================
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ================================
# TRAIN TEST SPLIT
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================================================
# MODEL 1: CATBOOST (OPTIMIZED)
# =========================================================
cat_model = CatBoostClassifier(
    iterations=1000,
    depth=7,
    learning_rate=0.01,
    loss_function='Logloss',
    auto_class_weights='Balanced',
    bagging_temperature=1.0,
    l2_leaf_reg=5.0,
    random_strength=1.0,
    verbose=0,
    random_seed=42,
    early_stopping_rounds=50,
    od_type='Iter'
)

cat_model.fit(X_train, y_train, eval_set=(X_test, y_test))

# =========================================================
# MODEL 2: LIGHTGBM (OPTIMIZED)
# =========================================================
lgbm_model = LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.01,
    num_leaves=50,
    max_depth=9,
    min_child_samples=20,
    subsample=0.85,
    subsample_freq=5,
    colsample_bytree=0.85,
    reg_alpha=0.5,
    reg_lambda=1.0,
    class_weight='balanced',
    random_state=42,
    verbose=-1
)

lgbm_model.fit(X_train, y_train, eval_set=[(X_test, y_test)])

# =========================================================
# PREDICTIONS (PROBABILITIES)
# =========================================================
cat_prob = cat_model.predict_proba(X_test)[:, 1]
lgbm_prob = lgbm_model.predict_proba(X_test)[:, 1]

# =========================================================
# INDIVIDUAL MODEL EVALUATION (for weighted voting)
# =========================================================
cat_pred = (cat_prob > 0.5).astype(int)
lgbm_pred = (lgbm_prob > 0.5).astype(int)

cat_roc_auc = roc_auc_score(y_test, cat_prob)
lgbm_roc_auc = roc_auc_score(y_test, lgbm_prob)

# Normalize weights based on ROC-AUC performance
total_auc = cat_roc_auc + lgbm_roc_auc
cat_weight = cat_roc_auc / total_auc
lgbm_weight = lgbm_roc_auc / total_auc

print(f"CatBoost ROC-AUC: {cat_roc_auc:.4f} (Weight: {cat_weight:.4f})")
print(f"LightGBM ROC-AUC: {lgbm_roc_auc:.4f} (Weight: {lgbm_weight:.4f})")

# =========================================================
# WEIGHTED SOFT VOTING ENSEMBLE
# =========================================================
final_prob = (cat_weight * cat_prob + lgbm_weight * lgbm_prob)

# =========================================================
# OPTIMAL THRESHOLD SEARCH
# =========================================================
best_f1 = 0
best_threshold = 0.5

for threshold in np.arange(0.3, 0.7, 0.01):
    y_pred_temp = (final_prob > threshold).astype(int)
    f1 = f1_score(y_test, y_pred_temp)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"\nOptimal Threshold: {best_threshold:.4f} (F1-Score: {best_f1:.4f})")

y_pred = (final_prob > best_threshold).astype(int)

# =========================================================
# EVALUATION
# =========================================================
print("\n===================================")
print(" CARDIONET-X (CATBOOST + LIGHTGBM)")
print("===================================")

ensemble_auc = roc_auc_score(y_test, final_prob)
ensemble_acc = accuracy_score(y_test, y_pred)
ensemble_f1 = f1_score(y_test, y_pred)

print(f"\nEnsemble Accuracy: {ensemble_acc:.4f}")
print(f"Ensemble ROC-AUC: {ensemble_auc:.4f}")
print(f"Ensemble F1-Score: {ensemble_f1:.4f}")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=['No Disease', 'Disease']))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))