import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import shap
import joblib

# --- STEP 1: Generate Synthetic "Financial" Data ---
# In a real project, you would load a CSV here (e.g., pd.read_csv('lending_club.csv'))
def generate_data(n_samples=5000):
    np.random.seed(42)
    
    # 1. Income (Annual)
    income = np.random.normal(60000, 15000, n_samples)
    
    # 2. Loan Amount
    loan_amount = np.random.normal(15000, 10000, n_samples)
    
    # 3. FICO Score (300-850)
    fico = np.random.normal(700, 50, n_samples)
    fico = np.clip(fico, 300, 850)
    
    # 4. Debt-to-Income Ratio (DTI) - Lower is better
    dti = np.random.normal(0.20, 0.1, n_samples)
    dti = np.clip(dti, 0, 1) # Cap between 0% and 100%

    # Create a DataFrame
    df = pd.DataFrame({
        'Income': income,
        'Loan_Amount': loan_amount,
        'FICO_Score': fico,
        'DTI': dti
    })
    
    # Define Logic for "Default" (Target Variable)
    # If FICO is low and DTI is high, higher chance of default (1)
    df['prob_default'] = 0.05 # Base risk
    df.loc[df['FICO_Score'] < 650, 'prob_default'] += 0.4
    df.loc[df['DTI'] > 0.4, 'prob_default'] += 0.3
    
    # Assign target based on probability
    df['Default'] = np.random.rand(n_samples) < df['prob_default']
    df['Default'] = df['Default'].astype(int)
    
    return df.drop(columns=['prob_default'])

print("Generating Data...")
df = generate_data()
## Step 1: Initial Data Exploration
## 1. View the first 5 rows (The 'head')
print("\n--- 1. First 5 Rows of Synthetic Data (The Head) ---")
print(df.head())

## 2. Check Data Types and Missing Values
print("\n--- 2. Data Info (Types and Nulls) ---")
print(df.info())

## 3. Check for Class Imbalance (How many Defaults vs. Pays)
print("\n--- 3. Class Balance Check (The Problem We Need to Solve) ---")
print(df['Default'].value_counts(normalize=True))
# The output here should show that '0' (Paid) is around 90% and '1' (Default) is around 10%.

## 4. Basic Descriptive Statistics
print("\n--- 4. Descriptive Statistics ---")
print(df.describe().T) # The .T transposes the table for better readability
## --- STEP 2: Preprocessing ---


# --- STEP 2: Preprocessing ---
X = df.drop('Default', axis=1)
y = df['Default']

# Split Data (80% Training, 20% Testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- STEP 3: Modeling with XGBoost ---
# scale_pos_weight is CRITICAL for financial data (handles imbalanced classes)
ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)

model = xgb.XGBClassifier(
    objective='binary:logistic',
    scale_pos_weight=ratio, 
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    use_label_encoder=False,
    eval_metric='logloss'
)

print("Training Model...")
model.fit(X_train, y_train)

# --- STEP 4: Evaluation ---
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n--- Model Performance ---")
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# --- STEP 5: Explainability (SHAP) ---
print("\nCalculating SHAP values...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# --- STEP 6: Save for the App ---
# Save the model and the explainer to use in the dashboard later
joblib.dump(model, 'C:\\Users\\Hp\\Documents\\Work project\\Smart-Lender Micro-Credit Risk Engine\\credit_risk_model.pkl')
joblib.dump(explainer, 'C:\\Users\\Hp\\Documents\\Work project\\Smart-Lender Micro-Credit Risk Engine\\shap_explainer.pkl')
print("\nModel and Explainer saved successfully!")