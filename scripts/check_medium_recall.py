import joblib
from sklearn.metrics import classification_report
import os

# -----------------------------
# 1️⃣ Set paths
# -----------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_DIR = os.path.join(BASE_DIR, 'models')

RF_MODEL_PATH = os.path.join(MODEL_DIR, 'random_forest_model.pkl')
XGB_MODEL_PATH = os.path.join(MODEL_DIR, 'xgboost_model.pkl')
LE_PATH = os.path.join(MODEL_DIR, 'label_encoder.pkl')
TEST_DATA_PATH = os.path.join(BASE_DIR, 'data/landslide_labeled_dataset.csv')  # full dataset

# -----------------------------
# 2️⃣ Load models and label encoder
# -----------------------------
rf_model = joblib.load(RF_MODEL_PATH)
xgb_model = joblib.load(XGB_MODEL_PATH)
le = joblib.load(LE_PATH)

# -----------------------------
# 3️⃣ Load test dataset
# -----------------------------
import pandas as pd
df = pd.read_csv(TEST_DATA_PATH)
df.columns = df.columns.str.strip()

features = ['Rainfall_mm', 'Slope_Angle', 'Soil_Saturation', 'Vegetation_Cover']
target = 'Risk_Label'

X = df[features]
y = le.transform(df[target])  # encode labels

# -----------------------------
# 4️⃣ Predictions
# -----------------------------
rf_pred = rf_model.predict(X)
xgb_pred = xgb_model.predict(X)

# -----------------------------
# 5️⃣ Classification report
# -----------------------------
rf_report = classification_report(y, rf_pred, target_names=le.classes_, output_dict=True)
xgb_report = classification_report(y, xgb_pred, target_names=le.classes_, output_dict=True)

# -----------------------------
# 6️⃣ Print Medium class recall
# -----------------------------
print("✅ Medium Class Recall Check\n")
print(f"Random Forest Medium Recall: {rf_report['Medium']['recall']:.3f}")
print(f"XGBoost Medium Recall: {xgb_report['Medium']['recall']:.3f}\n")

# -----------------------------
# 7️⃣ Optional: Overall summary
# -----------------------------
print("Random Forest F1-score (weighted):", rf_report['weighted avg']['f1-score'])
print("XGBoost F1-score (weighted):", xgb_report['weighted avg']['f1-score'])
