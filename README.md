# Fraud Detection Web App

This is a Streamlit web application for credit card fraud detection using machine learning.

## Features
- Train and evaluate ML models
- Simulate live transactions
- SHAP explainability plots
- User authentication (optional)
- Export filtered data

## Requirements
Install dependencies with:

```bash
pip install -r requirements.txt
```

## Run the App

```bash
streamlit run fraud-detection.py
```

## Deployment
For deployment on Streamlit Cloud:
- Ensure `requirements.txt` and `runtime.txt` are in the root folder.
- Add a `.streamlit/config.toml` for theme customization (optional).