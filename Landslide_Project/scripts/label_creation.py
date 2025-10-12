import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

# Load dataset
df = pd.read_csv("data/landslide_dataset.csv")

# Strip any extra spaces from column names
df.columns = df.columns.str.strip()

# Print columns to choose correct features
print("Columns in dataset:", df.columns)

# Update these according to your dataset columns
features = ['Rainfall_mm', 'Slope_Angle', 'Soil_Saturation', 'Vegetation_Cover']

# Make sure all features exist in df
for f in features:
    if f not in df.columns:
        raise ValueError(f"Column '{f}' not found in dataset. Check column names!")

# Normalize features
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df[features]), columns=features)

# Invert vegetation (more vegetation = less risk)
df_scaled['Vegetation_Cover'] = 1 - df_scaled['Vegetation_Cover']

# Risk Score (average of features)
df_scaled['Risk_Score'] = df_scaled.sum(axis=1) / len(features)

# Assign Risk Labels
def assign_risk_label(score):
    if score >= 0.7:
        return 'High'
    elif score >= 0.4:
        return 'Medium'
    else:
        return 'Low'

df['Risk_Label'] = df_scaled['Risk_Score'].apply(assign_risk_label)

# Check distribution
print(df['Risk_Label'].value_counts())

# Save labeled dataset
df.to_csv("data/landslide_labeled_dataset.csv", index=False)

# Save scaler for future predictions
joblib.dump(scaler, "models/scaler.pkl")

print("✅ Labeled dataset and scaler saved successfully!")
