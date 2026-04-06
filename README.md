LOANGUARD AI -- SMART CREDIT RISK ASSESSMENT

Project Overview:
LoanGuard AI is a real-time, explainable machine learning application designed to predict loan default probability with high precision. Built for financial analysts, the platform transforms complex "black-box" XGBoost predictions into clear, actionable insights using SHAP (SHapley Additive exPlanations).

Live Application:
loanguardai.streamlit.app

Key Features:

Predictive Intelligence: Leverages an optimized XGBoost model trained on 150,000+ borrower records.

Explainable AI (XAI): Interactive Waterfall and Bar charts reveal the exact financial drivers behind every risk score.

Custom Feature Engineering: Implements 25 advanced features including the Struggle Index, Weighted Late Scores, and Debt-to-Income Interactions.

Modern Financial Dashboard: A sleek, dark-mode UI optimized for real-time analysis and decision-making.

Audit-Ready Documentation: Integrated technical reports covering the entire data science lifecycle (Preprocessing, EDA, and Model Selection).

Tech Stack:

Languages: Python 3.11+

Frameworks: Streamlit (UI/Deployment), XGBoost (Modeling)

Explainability: SHAP (Step-by-step impact analysis)

Data Science: Pandas, NumPy, Scikit-learn, Joblib

Visualization: Matplotlib, Seaborn

Project Structure:
-- streamlit_app.py        (Main Streamlit dashboard application)
-- 1_Project.py            (Model training and feature engineering script)
-- best_xgb_model.joblib   (Serialized XGBoost model)
-- data_scaler.joblib      (Serialized RobustScaler for data normalization)
-- reports/                (Technical markdown documentation)
-- requirements.txt        (Production dependencies)

Model Methodology:
The model is trained on the Kaggle "Give Me Some Credit" dataset.

Objective: Maximize Recall to capture 75%+ of actual defaulters while maintaining a balanced Precision/F1 score.

Handling Imbalance: Optimized using scale_pos_weight and threshold tuning (set at 0.1883) to address the 6.6% minority class distribution.

Feature Engineering: Validated that engineered features like Youth_Utilization_Risk provide higher predictive value than raw data alone.

Local Development:

Clone the Repository:
git clone [Your GitHub URL]
cd LoanDefault

Install Dependencies:
pip install -r requirements.txt

Run the App:
streamlit run streamlit_app.py

License:
This project is distributed under the MIT License.
