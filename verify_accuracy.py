import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
import os

try:
    # Load data
    test_df = pd.read_csv("data/test.csv")
    X_test = test_df.drop(columns=['churn'])
    y_test = test_df['churn']

    # Load model
    model = joblib.load("model/model.joblib")

    # Predict
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)

    print(f"Model Accuracy: {accuracy}")
except Exception as e:
    print(f"Error: {e}")
