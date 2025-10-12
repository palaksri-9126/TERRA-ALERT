import joblib
import numpy as np
import os

# -----------------------------
# 1️⃣ Set paths relative to this script
# -----------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_DIR = os.path.join(BASE_DIR, 'models')

RF_MODEL_PATH = os.path.join(MODEL_DIR, 'random_forest_model.pkl')
XGB_MODEL_PATH = os.path.join(MODEL_DIR, 'xgboost_model.pkl')
LE_PATH = os.path.join(MODEL_DIR, 'label_encoder.pkl')

# -----------------------------
# 2️⃣ Load models and label encoder
# -----------------------------
rf_model = joblib.load(RF_MODEL_PATH)
xgb_model = joblib.load(XGB_MODEL_PATH)
le = joblib.load(LE_PATH)

# -----------------------------
# 3️⃣ Example input for prediction
# Replace these values with new test data
# -----------------------------
# Features order must match training features
# Example: ['Rainfall', 'Slope_Angle', 'Soil_Type', 'Vegetation']
new_input = np.array([[1200, 30, 2, 0.3]])  # Example values

# -----------------------------
# 4️⃣ Make predictions
# -----------------------------
rf_pred_numeric = rf_model.predict(new_input)
xgb_pred_numeric = xgb_model.predict(new_input)

# Decode numeric predictions back to original labels
rf_pred_label = le.inverse_transform(rf_pred_numeric)
xgb_pred_label = le.inverse_transform(xgb_pred_numeric)

# -----------------------------
# 5️⃣ Print predictions
# -----------------------------
print("Random Forest Prediction:", rf_pred_label[0])
print("XGBoost Prediction:", xgb_pred_label[0])
