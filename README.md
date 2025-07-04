# 🏦 Smart Loan Approval & Fraud Detection System

**Smart Loan Approval & Fraud Detection System** is an interactive tool that helps financial institutions make smarter lending decisions. By analyzing applicant details and transaction behavior, the system predicts whether a loan should be approved and flags potentially fraudulent applications. It features a user-friendly interface for real-time predictions and provides insights into the key factors influencing each decision.

---

## ✨ Features

- Interactive UI with **Streamlit**
- Real-time **loan approval prediction**
- Fraud detection using transaction behavior
- **Model explainability** with SHAP
- Visual analysis using **Seaborn** and **Matplotlib**
- Machine learning using **Scikit-learn** and **XGBoost**

---

## 📊 Data Description

- **Loan Application Data**: Includes applicant demographics, income, loan purpose, credit score, etc.
- **Transaction Data**: Includes customer transaction types, frequency, amount, failure rate, and international transaction count.

The two datasets are merged using `customer_id` to create a consolidated training dataset.

---

## 🔁 Workflow

1. Data Loading
2. Data Cleaning & Preprocessing
3. Feature Engineering
4. Model Training (Logistic Regression, XGBoost)
5. Evaluation (Confusion Matrix, ROC, AUC)
6. Explainability using SHAP
7. Deployment via Streamlit

---

## ⚙️ Installation

```bash
# Step 1: Clone the repository
git clone https://github.com/karnika-c-s/smart_loan_approval_and_fraud-_detection.git
cd smart_loan_approval_and_fraud-_detection

# Step 2: Install dependencies
pip install -r requirements.txt

# Step 3: Run the Streamlit app
streamlit run app.py
