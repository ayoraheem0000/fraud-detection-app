#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Load Dataset
df = pd.read_csv('C:\\Users\\araheem\\Desktop\\fraud-detection-app\\creditcard.csv')

# Optional: Downsample for speed (remove for full data)
df = pd.concat([
    df[df['Class'] == 0].sample(10000, random_state=42),
    df[df['Class'] == 1]
])

# Split features and target
X = df.drop(columns=['Class'])
y = df['Class']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

# Train Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save Model
joblib.dump(model, 'fraud_model.pkl')
print("Model saved as 'fraud_model.pkl'")

# Load Model
model = joblib.load('fraud_model.pkl')
print("Model loaded successfully.")

# Evaluate Model
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# SHAP Explainability
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test[:100])  # Limit to 100 rows for speed

# Summary Plot
shap.summary_plot(shap_values, X_test[:100])

# Force Plot for a single prediction
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[0].values, X_test.iloc[0, :], matplotlib=True)

# Simulate Live Transaction
def simulate_transaction():
    idx = np.random.randint(0, len(X_test))
    sample = X_test.iloc[idx:idx+1]
    prediction = model.predict(sample)[0]
    prob = model.predict_proba(sample)[0][1]

    print(f"\nSimulated Transaction Index: {idx}")
    print(f"Prediction: {'FRAUD' if prediction == 1 else 'LEGITIMATE'}")
    print(f"Probability of Fraud: {prob:.4f}")

    # SHAP explanation
    shap_values_sim = explainer(sample)
    shap.plots.waterfall(shap_values_sim[0])

simulate_transaction()

