# 🌍 Terra-Alert
**Predict. Prevent. Protect.**

Terra-Alert is an AI-powered landslide prediction system that analyzes environmental factors to estimate the risk of landslides. The system uses machine learning models to help identify vulnerable regions and support early warning systems for disaster management.

## 🚀 Features
- AI-based landslide risk prediction
- Uses environmental parameters like rainfall, slope, soil saturation, and vegetation cover
- Classifies regions into **High, Medium, and Low risk zones**
- Interactive dashboard for real-time predictions
- Data-driven insights for disaster prevention

## 🛠 Tech Stack
- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Machine Learning (Random Forest, XGBoost)

## 📂 Project Structure
```
Terra-Alert/
│
├── data/                # Dataset used for training
├── models/              # Trained ML models
├── app.py               # Streamlit application
├── preprocessing.py     # Data preprocessing
├── train_model.py       # Model training script
├── requirements.txt     # Project dependencies
└── README.md
```

## ⚙️ Installation

Clone the repository
```bash
git clone https://github.com/your-username/terra-alert.git
```

Navigate to the project directory
```bash
cd terra-alert
```

Install dependencies
```bash
pip install -r requirements.txt
```

Run the application
```bash
streamlit run app.py
```

## 📊 How It Works
1. User inputs environmental parameters.
2. The system processes the data using trained ML models.
3. The model predicts landslide risk levels.
4. Results are displayed on the dashboard.

## 🔮 Future Improvements
- Integration with real-time weather data
- Automated alert system for authorities
- Mobile notifications for disaster warnings
- Expansion to other disaster predictions like floods and droughts

## 📜 License
This project is developed for educational and research purposes.
