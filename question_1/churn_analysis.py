# Telco Customer Churn Analysis - Complete Script

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, ttest_ind
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_auc_score)

# 1. Load Data
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# 2. Data Preprocessing
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()

# 3. Descriptive Statistics
print("Descriptive Statistics for tenure, MonthlyCharges, TotalCharges:")
print(df[['tenure', 'MonthlyCharges', 'TotalCharges']].describe())
print("\nChurn distribution:")
print(df['Churn'].value_counts())

# 4. Churn Rates Across Contract Type & Internet Service
pd.crosstab(df['Contract'], df['Churn']).plot(kind='bar', stacked=True)
plt.title("Churn by Contract Type")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

pd.crosstab(df['InternetService'], df['Churn']).plot(kind='bar', stacked=True)
plt.title("Churn by Internet Service")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# 5. Inferential Statistics
print("\nChi-Square Test Results:")
for col in ['InternetService', 'Contract', 'SeniorCitizen']:
    tab = pd.crosstab(df[col], df['Churn'])
    _, p, _, _ = chi2_contingency(tab)
    print(f'{col} vs Churn: p-value = {p:.4f}')

print("\nT-Test Results:")
for col in ['MonthlyCharges', 'TotalCharges']:
    churned = df[df['Churn'] == 'Yes'][col]
    not_churned = df[df['Churn'] == 'No'][col]
    t_stat, p_val = ttest_ind(churned, not_churned)
    print(f'{col}: p-value = {p_val:.4f}')

# 6. Prepare Data for Modeling
X = df.drop(['customerID', 'Churn'], axis=1)
X = pd.get_dummies(X)
y = df['Churn'].map({'No': 0, 'Yes': 1})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# 7. Logistic Regression Model
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

print('\nLogistic Regression Results:')
print(classification_report(y_test, y_pred_lr))
print('ROC-AUC Score:', roc_auc_score(y_test, lr.predict_proba(X_test)[:,1]))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred_lr))

# 8. Decision Tree Model
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

print('\nDecision Tree Results:')
print(classification_report(y_test, y_pred_dt))
print('ROC-AUC Score:', roc_auc_score(y_test, dt.predict_proba(X_test)[:,1]))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred_dt))

# 9. Feature Importance
coefs = lr.coef_[0]
lr_idx = np.argsort(np.abs(coefs))[-5:]
print("\nTop 5 features (Logistic Regression):")
for idx in lr_idx:
    print(X.columns[idx])

importances = dt.feature_importances_
dt_idx = np.argsort(importances)[-5:]
print("\nTop 5 features (Decision Tree):")
for idx in dt_idx:
    print(X.columns[idx])

# 10. Correlation Heatmap
plt.figure(figsize=(12, 8))
corr = X.corr()
sns.heatmap(corr, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()
