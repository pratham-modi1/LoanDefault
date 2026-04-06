🛡️ LoanGuard AI — Smart Credit Risk Assessment
LoanGuard AI is a high-performance, explainable machine learning dashboard designed to predict the probability of loan default. Unlike traditional "black-box" models, LoanGuard leverages SHAP (SHapley Additive exPlanations) to provide transparent, feature-level insights for every prediction.

🚀 Live Demo
Check out the live application here: loanguardai.streamlit.app

✨ Key Features
Real-time Prediction: Instant default probability scoring using an optimized XGBoost pipeline.

Explainable AI (XAI): Interactive Waterfall and Bar plots reveal exactly why a borrower was flagged as high-risk.

Advanced Feature Engineering: Includes custom-built indicators like the Struggle Index, Weighted Late Scores, and Debt-to-Income Interactions.

Sleek UI/UX: A custom-styled dark mode interface built with Streamlit, optimized for financial analysts.

Technical Reporting: Integrated documentation covering Preprocessing, EDA, and Model Selection.

🛠️ Tech Stack
Core: Python 3.11+

Machine Learning: XGBoost, Scikit-learn

Explainability: SHAP (SHapley Additive exPlanations)

Deployment & UI: Streamlit

Data Handling: Pandas, NumPy

Visualization: Matplotlib, Seaborn

📂 Project Structure
Plaintext
├── streamlit_app.py        # Main Streamlit application logic & UI
├── 1_Project.py            # Model training & feature engineering pipeline
├── best_xgb_model.joblib   # Pre-trained XGBoost model
├── data_scaler.joblib      # Fitted RobustScaler for input normalization
├── reports/                # Markdown files for the technical documentation tab
├── utility/                # Data dictionaries and evaluation logs
└── requirements.txt        # Production dependencies
⚙️ Local Setup
To run this project locally, follow these steps:

Clone the repository:

Bash
git clone https://github.com/your-username/LoanDefault.git
cd LoanDefault
Create a virtual environment:

Bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies:

Bash
pip install -r requirements.txt
Launch the App:

Bash
streamlit run streamlit_app.py
🧠 Model Methodology
The underlying model was trained on the Kaggle "Give Me Some Credit" dataset (150,000+ records).

Optimization: Hyperparameter tuning via GridSearchCV focused on maximizing Recall (catching 75% of actual defaulters).

Handling Imbalance: Applied scale_pos_weight to manage the high class-imbalance ratio (6.6% default rate).
