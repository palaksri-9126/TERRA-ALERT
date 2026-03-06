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
terra-alert/
│
├── data/                # Dataset used for training and testing
├── models/              # Saved machine learning models
├── scripts/             # Data preprocessing and model training scripts
│
├── streamlit_app.py     # Streamlit web application
├── .gitignore           # Git ignored files configuration
├── LICENSE              # Project license
└── README.md            # Project documentation
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
pip install streamlit pandas numpy scikit-learn matplotlib
```

Run the application
```bash
streamlit run streamlit_app.py
```

## 📸 Project Screenshots

<p align="center">
  <img width="45%" alt="Screenshot 2026-03-06 235449" src="https://github.com/user-attachments/assets/59dea127-4e4a-4039-bbba-5e3be5470c11" />
  <img width="45%" alt="Screenshot 2026-03-06 235513" src="https://github.com/user-attachments/assets/16fc3225-066e-47a9-bbe0-9b06fe7b0621" />
</p>
<p align="center">
  <img width="45%" alt="Screenshot 2026-03-06 235520" src="https://github.com/user-attachments/assets/58a0d7a4-9ffe-46d2-9a36-c77cb9618e03" />
  <img width="45%" alt="Screenshot 2026-03-06 235528" src="https://github.com/user-attachments/assets/ae7d927a-b67c-4950-94fe-6eddaf12f05d" />
</p>

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
