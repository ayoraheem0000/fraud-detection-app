import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
)
import io

st.set_page_config(page_title="Fraud Detection App", layout="wide")
st.title("💳 Fraud Detection Web App")

@st.cache_data
def generate_data(n_samples=1000):
    np.random.seed(42)
    transaction_amount = np.random.exponential(scale=100, size=n_samples)
    transaction_time = np.random.randint(0, 24, size=n_samples)
    user_behavior_score = np.random.normal(loc=0.5, scale=0.15, size=n_samples)
    fraud = ((transaction_amount > 180) & (user_behavior_score < 0.4)).astype(int)
    return pd.DataFrame({
        'TransactionAmount': transaction_amount,
        'TransactionTime': transaction_time,
        'UserBehaviorScore': user_behavior_score,
        'IsFraud': fraud
    })

df = generate_data()

@st.cache_resource
def train_model(data):
    X = data[['TransactionAmount', 'TransactionTime', 'UserBehaviorScore']]
    y = data['IsFraud']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model, X_test, y_test

model, X_test, y_test = train_model(df)

# Sidebar for single prediction
st.sidebar.header("🔍 Single Transaction Prediction")
amount = st.sidebar.number_input("Transaction Amount", min_value=0.0, value=100.0)
time = st.sidebar.slider("Transaction Time (Hour)", 0, 23, 12)
score = st.sidebar.slider("User Behavior Score", 0.0, 1.0, 0.5)

if st.sidebar.button("Predict Fraud"):
    input_data = pd.DataFrame([[amount, time, score]], columns=['TransactionAmount', 'TransactionTime', 'UserBehaviorScore'])
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    st.sidebar.success(f"Prediction: {'Fraud' if prediction else 'Legit'} (Probability: {probability:.2f})")

# Tabs
tab1, tab2, tab3 = st.tabs(["📁 Batch Prediction", "📊 Metrics", "📈 Visualizations"])

with tab1:
    st.header("📁 Upload CSV for Batch Prediction")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        try:
            batch_data = pd.read_csv(uploaded_file)
            required_cols = ['TransactionAmount', 'TransactionTime', 'UserBehaviorScore']
            if all(col in batch_data.columns for col in required_cols):
                preds = model.predict(batch_data[required_cols])
                probs = model.predict_proba(batch_data[required_cols])[:, 1]
                batch_data['IsFraudPrediction'] = preds
                batch_data['FraudProbability'] = probs
                st.dataframe(batch_data)

                csv_buffer = io.StringIO()
                batch_data.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="📥 Download Predictions as CSV",
                    data=csv_buffer.getvalue(),
                    file_name="fraud_predictions.csv",
                    mime="text/csv"
                )
            else:
                st.error(f"CSV must contain columns: {required_cols}")
        except Exception as e:
            st.error(f"Error reading file: {e}")

with tab2:
    st.header("📊 Model Evaluation Metrics")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    st.write(f"Precision: {precision_score(y_test, y_pred):.2f}")
    st.write(f"Recall: {recall_score(y_test, y_pred):.2f}")
    st.write(f"F1 Score: {f1_score(y_test, y_pred):.2f}")
    st.write(f"ROC AUC Score: {roc_auc_score(y_test, y_prob):.2f}")

with tab3:
    st.header("📈 Visualizations")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Legit', 'Fraud'], yticklabels=['Legit', 'Fraud'], ax=ax_cm)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    ax_cm.set_title("Confusion Matrix")
    st.pyplot(fig_cm)

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_prob):.2f}")
    ax_roc.plot([0, 1], [0, 1], 'k--')
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curve")
    ax_roc.legend()
    st.pyplot(fig_roc)

    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    fig_pr, ax_pr = plt.subplots()
    ax_pr.plot(recall, precision)
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall Curve")
    st.pyplot(fig_pr)
