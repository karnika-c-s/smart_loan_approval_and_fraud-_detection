#IMPORTING PACKAGES
 
# Data manipulation & visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Explainability
import shap

# Warnings
import warnings
warnings.filterwarnings("ignore")

# Display settings
pd.set_option('display.max_columns', None)

# Loading loan applications data
loan_df = pd.read_csv('../data/loan_applications.csv')

# Preview both datasets
print("Loan Applications Data:")
print(loan_df.head(3))

# Loading transaction data
txn_df=pd.read_csv('../data/transactions.csv')

print("\nTransaction Data:")
print(txn_df.head(3))

# Shape and data types
print("Shape of loan application data:", loan_df.shape)
loan_df.info()
print("Shape of Transaction data:", txn_df.shape)
txn_df.info()

# Check missing values
print("\nMissing values in loan application data:")
loan_df.isnull().sum()

# Check for missing values
print("\nMissing values transaction data:")
txn_df.isnull().sum()

# Loan Status Distribution
sns.countplot(x='loan_status', data=loan_df)
plt.title("Loan Approval Status Count")
plt.show()

# Fraud Flag Distribution
sns.countplot(x='fraud_flag', data=loan_df)
plt.title("Fraud Flag Distribution")
plt.show()

# Age, Income, Loan Amount
loan_df[['applicant_age', 'monthly_income', 'loan_amount_requested']].describe()

# Distribution Plots
fig, axs = plt.subplots(1, 3, figsize=(18, 4))
sns.histplot(loan_df['applicant_age'], kde=True, ax=axs[0])
sns.histplot(loan_df['monthly_income'], kde=True, ax=axs[1])
sns.histplot(loan_df['loan_amount_requested'], kde=True, ax=axs[2])
axs[0].set_title("Applicant Age")
axs[1].set_title("Monthly Income")
axs[2].set_title("Loan Amount Requested")
plt.show()

plt.figure(figsize=(6, 3))
sns.countplot(x='transaction_status', data=txn_df)
plt.title("Transaction Status (Success vs Failure)")
plt.show()


# Convert transaction_date to datetime
txn_df['transaction_date'] = pd.to_datetime(txn_df['transaction_date'], dayfirst=True)

# Feature engineering - summarize transaction behavior per customer
txn_summary = txn_df.groupby('customer_id').agg({
    'transaction_amount': ['mean', 'sum', 'count'],
    'transaction_type': lambda x: x.mode()[0] if not x.mode().empty else np.nan,
    'device_used': lambda x: x.mode()[0] if not x.mode().empty else np.nan,
    'transaction_status': lambda x: (x != 'Success').mean(),
    'is_international_transaction': 'sum'
}).reset_index()

# Rename columns for clarity
txn_summary.columns = ['customer_id', 'avg_txn_amount', 'total_txn_amount', 'txn_count',
                       'most_common_txn_type', 'most_used_device', 'failed_txn_ratio', 'intl_txn_count']

# Preview
txn_summary.head()

# Merge txn_summary into loan_df using customer_id
loan_merged = pd.merge(loan_df, txn_summary, on='customer_id', how='left')

# Show merged result
print("Merged Data Shape:", loan_merged.shape)
loan_merged.head(3)

# Copy to avoid changing original
df = loan_merged.copy()

# Handle missing values

# Fill numerical txn columns with 0 (e.g., no transactions)
txn_cols = ['avg_txn_amount', 'total_txn_amount', 'txn_count', 'failed_txn_ratio', 'intl_txn_count']
df[txn_cols] = df[txn_cols].fillna(0)

# Fill categorical with "Unknown"
df['most_common_txn_type'] = df['most_common_txn_type'].fillna("Unknown")
df['most_used_device'] = df['most_used_device'].fillna("Unknown")

# Encode categorical columns
cat_cols = ['loan_type', 'purpose_of_loan', 'employment_status', 'property_ownership_status',
            'gender', 'most_common_txn_type', 'most_used_device']

le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))

# Drop unused columns (IDs, text-heavy, address)
drop_cols = ['application_id', 'customer_id', 'application_date', 'residential_address', 'fraud_type', 'transaction_notes']
df = df.drop(columns=[col for col in drop_cols if col in df.columns])

# Convert target labels
df['loan_status_binary'] = df['loan_status'].map({
    'Approved': 0,  # or 1 if you prefer — just be consistent
    'Declined': 1,
    'Fraudulent - Detected': 1,
    'Fraudulent - Undetected': 1
})

df['fraud_flag'] = df['fraud_flag'].astype(int)

import seaborn as sns
import matplotlib.pyplot as plt
numeric_cols = df.select_dtypes(include='number')
plt.figure(figsize=(12, 8))
sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap (Merged Features)")
plt.show()

from sklearn.model_selection import train_test_split
# Replace loan_status with binary labels
df['loan_status_binary'] = df['loan_status'].replace({
    'Approved': 0,
    'Declined': 1,
    'Fraudulent - Detected': 1,
    'Fraudulent - Undetected': 1
})

# Drop rows where loan_status_binary is still NaN
df = df.dropna(subset=['loan_status_binary'])

X = df.drop(['loan_status', 'fraud_flag', 'loan_status_binary'], axis=1)
y = df['loan_status_binary']

# Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

# Separate numeric and categorical columns
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# Preprocessing
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine into ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

clf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(solver='lbfgs', max_iter=1000))
])

clf_pipeline.fit(X_train, y_train)

from sklearn.metrics import classification_report, confusion_matrix

y_pred = clf_pipeline.predict(X_test)

print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

import joblib
joblib.dump(clf_pipeline, 'loan_classifier_binary.pkl')

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

y_probs = clf_pipeline.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend()
plt.show()

# Get feature names after transformation
feature_names = clf_pipeline.named_steps['preprocessor'].get_feature_names_out()

# Get coefficients from the logistic regression model
coef = clf_pipeline.named_steps['classifier'].coef_[0]

# Create DataFrame to show feature importance
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': coef
}).sort_values(by='Importance', key=abs, ascending=False)

# Display top 10 important features
print(importance_df.head(10))

import matplotlib.pyplot as plt

# Plot top 10 features
top_n = 10
top_features = importance_df.head(top_n)

plt.figure(figsize=(10, 6))
plt.barh(top_features['Feature'], top_features['Importance'], color='skyblue')
plt.xlabel("Coefficient (Importance)")
plt.title(f"Top {top_n} Important Features from Logistic Regression")
plt.gca().invert_yaxis()  # Highest on top
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

clf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        scale_pos_weight=4.5,  # imbalance: 40882/9118
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    ))
])

clf_pipeline.fit(X_train, y_train)

y_pred = clf_pipeline.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

from sklearn.metrics import  roc_auc_score

y_pred = clf_pipeline.predict(X_test)
y_prob = clf_pipeline.predict_proba(X_test)[:, 1]
print("ROC AUC Score:", roc_auc_score(y_test, y_prob))

