import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, f1_score
from imblearn.over_sampling import SMOTE
import joblib
import os

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
DATA_PATH = os.path.join(BASE_DIR, 'data/landslide_labeled_dataset.csv')

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

# -----------------------------
# Base features
# -----------------------------
features = ['Rainfall_mm', 'Slope_Angle', 'Soil_Saturation', 'Vegetation_Cover']
X = df[features]
y = df['Risk_Label']

# -----------------------------
# Feature engineering
# -----------------------------
X['Rainfall_Slope'] = X['Rainfall_mm'] * X['Slope_Angle']
X['Rainfall_SoilRatio'] = X['Rainfall_mm'] / (X['Soil_Saturation'] + 1e-6)
X['Slope_Soil'] = X['Slope_Angle'] * X['Soil_Saturation']

engineered_features = features + ['Rainfall_Slope', 'Rainfall_SoilRatio', 'Slope_Soil']
X = X[engineered_features]

# -----------------------------
# Encode target
# -----------------------------
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print("Classes mapping:", dict(zip(le.classes_, range(len(le.classes_)))))

# -----------------------------
# Handle class imbalance using SMOTE
# -----------------------------
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y_encoded)

# -----------------------------
# Split train/test
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42
)

# -----------------------------
# Load trained models
# -----------------------------
rf_model = joblib.load(os.path.join(MODEL_DIR, "random_forest_model.pkl"))
xgb_model = joblib.load(os.path.join(MODEL_DIR, "xgboost_model.pkl"))

# -----------------------------
# Training vs Test Accuracy
# -----------------------------
rf_train_acc = accuracy_score(y_train, rf_model.predict(X_train))
rf_test_acc = accuracy_score(y_test, rf_model.predict(X_test))

xgb_train_acc = accuracy_score(y_train, xgb_model.predict(X_train))
xgb_test_acc = accuracy_score(y_test, xgb_model.predict(X_test))

print("\n=== Training vs Test Accuracy ===")
print(f"Random Forest -> Train: {rf_train_acc:.3f}, Test: {rf_test_acc:.3f}")
print(f"XGBoost -> Train: {xgb_train_acc:.3f}, Test: {xgb_test_acc:.3f}")

# -----------------------------
# Classification reports
# -----------------------------
rf_pred = rf_model.predict(X_test)
xgb_pred = xgb_model.predict(X_test)

print("\n=== Random Forest Classification Report ===")
print(classification_report(y_test, rf_pred, target_names=le.classes_))

print("\n=== XGBoost Classification Report ===")
print(classification_report(y_test, xgb_pred, target_names=le.classes_))

# -----------------------------
# 5-Fold Cross-Validation (weighted F1)
# -----------------------------
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rf_cv_scores = cross_val_score(rf_model, X_res, y_res, cv=kf, scoring='f1_weighted')
xgb_cv_scores = cross_val_score(xgb_model, X_res, y_res, cv=kf, scoring='f1_weighted')

print("\n=== 5-Fold Cross-Validation Weighted F1 ===")
print("Random Forest CV Scores:", rf_cv_scores)
print("Random Forest CV Mean:", rf_cv_scores.mean())
print("XGBoost CV Scores:", xgb_cv_scores)
print("XGBoost CV Mean:", xgb_cv_scores.mean())

# =============================================
# 🔹 Visual Performance Summary
# =============================================
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize

# -------------------------------
# Confusion Matrix - Random Forest
# -------------------------------
plt.figure(figsize=(6, 5))
ConfusionMatrixDisplay.from_estimator(rf_model, X_test, y_test, display_labels=le.classes_, cmap='Blues')
plt.title("Random Forest - Confusion Matrix")
plt.show()

# -------------------------------
# Confusion Matrix - XGBoost
# -------------------------------
plt.figure(figsize=(6, 5))
ConfusionMatrixDisplay.from_estimator(xgb_model, X_test, y_test, display_labels=le.classes_, cmap='Greens')
plt.title("XGBoost - Confusion Matrix")
plt.show()

# -------------------------------
# ROC Curves (multi-class)
# -------------------------------
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
y_score_rf = rf_model.predict_proba(X_test)
y_score_xgb = xgb_model.predict_proba(X_test)

plt.figure(figsize=(8, 6))
for i, class_name in enumerate(le.classes_):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score_rf[:, i])
    plt.plot(fpr, tpr, label=f"RF - {class_name} (AUC={auc(fpr, tpr):.2f})")
plt.title("Random Forest ROC Curves")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
for i, class_name in enumerate(le.classes_):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score_xgb[:, i])
    plt.plot(fpr, tpr, label=f"XGB - {class_name} (AUC={auc(fpr, tpr):.2f})")
plt.title("XGBoost ROC Curves")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

# -------------------------------
# Feature Importance
# -------------------------------
rf_importance = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
xgb_importance = pd.Series(xgb_model.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(8, 5))
rf_importance.head(10).plot(kind='barh', color='skyblue')
plt.title("Random Forest - Top 10 Feature Importances")
plt.gca().invert_yaxis()
plt.show()

plt.figure(figsize=(8, 5))
xgb_importance.head(10).plot(kind='barh', color='lightgreen')
plt.title("XGBoost - Top 10 Feature Importances")
plt.gca().invert_yaxis()
plt.show()
