import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import joblib
import os

# -----------------------------
# 1️⃣ Load labeled dataset
# -----------------------------
df = pd.read_csv("data/landslide_labeled_dataset.csv")
df.columns = df.columns.str.strip()  # remove extra spaces

# -----------------------------
# 2️⃣ Define base features
# -----------------------------
features = ['Rainfall_mm', 'Slope_Angle', 'Soil_Saturation', 'Vegetation_Cover']
for f in features:
    if f not in df.columns:
        raise ValueError(f"Column '{f}' not found in dataset!")

X = df[features]
y = df['Risk_Label']

# -----------------------------
# 3️⃣ Feature Engineering
# -----------------------------
# Derived / interaction features
X['Rainfall_Slope'] = X['Rainfall_mm'] * X['Slope_Angle']
X['Rainfall_SoilRatio'] = X['Rainfall_mm'] / (X['Soil_Saturation'] + 1e-6)
X['Slope_Soil'] = X['Slope_Angle'] * X['Soil_Saturation']

# Update features list to include engineered features
engineered_features = features + ['Rainfall_Slope', 'Rainfall_SoilRatio', 'Slope_Soil']
X = X[engineered_features]

print("Features used for training:", engineered_features)

# -----------------------------
# 4️⃣ Encode target labels
# -----------------------------
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print("Classes mapping:", dict(zip(le.classes_, range(len(le.classes_)))))

# -----------------------------
# 5️⃣ Handle class imbalance using SMOTE
# -----------------------------
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y_encoded)
print(f"Shape after SMOTE: {X_res.shape}, {y_res.shape}")

# -----------------------------
# 6️⃣ Split train/test
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42
)

# -----------------------------
# 7️⃣ Train Random Forest (with class_weight='balanced')
# -----------------------------
rf_model = RandomForestClassifier(
    n_estimators=200, max_depth=7, class_weight='balanced', random_state=42
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# -----------------------------
# 8️⃣ Train XGBoost (optimized to reduce overfitting)
# -----------------------------
xgb_model = XGBClassifier(
    n_estimators=300,             # thoda zyada trees, lekin learning_rate kam
    max_depth=4,                 # depth kam karke overfitting reduce
    learning_rate=0.05,          # chhoti LR for smooth learning
    subsample=0.7,               # random subset for generalization
    colsample_bytree=0.7,        # feature sampling to reduce correlation
    min_child_weight=3,          # prevent learning from small noisy leaf
    gamma=0.2,                   # minimum loss reduction for split
    reg_alpha=0.3,               # L1 regularization
    reg_lambda=1.2,              # L2 regularization
    random_state=42,
    eval_metric='mlogloss'
)

xgb_model.fit(X_train, y_train)

# Predict and evaluate
xgb_pred = xgb_model.predict(X_test)

# -----------------------------
# 9️⃣ Evaluation
# -----------------------------
print("\nRandom Forest Accuracy:", accuracy_score(y_test, rf_pred))
print("\nXGBoost Accuracy:", accuracy_score(y_test, xgb_pred))

print("\nRandom Forest Classification Report:\n", classification_report(y_test, rf_pred, target_names=le.classes_))
print("\nXGBoost Classification Report:\n", classification_report(y_test, xgb_pred, target_names=le.classes_))

# -----------------------------
# 10️⃣ Save models and label encoder
# -----------------------------
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(rf_model, os.path.join(MODEL_DIR, "random_forest_model.pkl"))
joblib.dump(xgb_model, os.path.join(MODEL_DIR, "xgboost_model.pkl"))
joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder.pkl"))

print("\n✅ Models and label encoder saved successfully!")
