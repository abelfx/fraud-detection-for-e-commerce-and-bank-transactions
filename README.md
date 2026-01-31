# 🛡️ Fraud Detection for E-Commerce and Banking
### *Advanced Machine Learning Solutions for Adey Innovations Inc.*

## 📌 Project Overview
As a Data Scientist at **Adey Innovations Inc.**, I am developing a robust fraud detection system for two distinct domains: **E-commerce transactions** and **Bank credit transactions**. 

Fraud detection is a high-stakes challenge involving a delicate trade-off between **Security** (detecting fraud) and **User Experience** (minimizing false positives). This project leverages geolocation analysis, transaction patterns, and advanced ensemble models to protect financial assets and build institutional trust.

---

## 📂 Project Structure
The repository is organized following industry best practices for data science workflows:

```bash
fraud-detection/
├── .vscode/                 # Editor settings
├── .github/workflows/       # CI/CD: Automated unittests
├── data/                    # Data storage (Added to .gitignore)
│   ├── raw/                 # Original datasets
│   └── processed/           # Cleaned and feature-engineered data
├── notebooks/               # Step-by-step analysis & modeling
│   ├── eda-fraud-data.ipynb
│   ├── eda-creditcard.ipynb
│   ├── feature-engineering.ipynb
│   ├── modeling.ipynb
│   └── shap-explainability.ipynb
├── src/                     # Modular source code
├── tests/                   # Unit tests for data integrity
├── models/                  # Saved model artifacts (.pkl, .joblib)
├── scripts/                 # Utility scripts for data processing
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

## 🛠️ Key Technical Challenges
- Class Imbalance: Fraud cases make up < 1% of the data. We utilize SMOTE (Synthetic Minority Over-sampling Technique) to ensure the model learns fraud patterns effectively.
- Geolocation Mapping: Merging billion-row IP ranges with transaction logs using range-based lookups (merge_asof) for country-level insights.
- Explainability: Using SHAP (SHapley Additive exPlanations) to move beyond "black-box" models and provide actionable business recommendations.

## 🚀 Installation & Setup
1. Clone the Repository:
```
git clone [https://github.com/your-username/fraud-detection.git](https://github.com/your-username/fraud-detection.git)
cd fraud-detection
```

2. Environment Setup:
```
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

4. Data Preparation:
Place Fraud_Data.csv, IpAddress_to_Country.csv, and creditcard.csv inside the data/raw/ folder.

# 📊 Pipeline Stages
## Task 1: Data Analysis & Preprocessing
- IP Mapping: Converted IP addresses to integers to perform range-based country lookups.

## Feature Engineering:

- time_diff: Duration between signup and purchase (critical for spotting "instant" bot-driven fraud).
- hour_of_day & day_of_week: Temporal patterns in fraudulent behavior.
- Imbalance Handling: Applied SMOTE only to the training set to prevent data leakage.

## Task 2: Model Building & Training
- Baseline Model: Logistic Regression for clear interpretability.
- Ensemble Models: Random Forest and XGBoost for capturing complex, non-linear fraud signatures.
- Evaluation Metrics: Priority given to AUC-PR and F1-Score over simple accuracy.

## Task 3: Model Explainability (XAI)
- Extracting Global Importance to identify overall fraud drivers.
- Generating SHAP Force Plots for individual transaction verification (True Positives vs. False Positives).


