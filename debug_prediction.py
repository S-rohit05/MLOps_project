import joblib
import pandas as pd
import os

# Load model
model_path = os.path.join("model", "model.joblib")
if not os.path.exists(model_path):
    print(f"Model not found at {model_path}")
    exit(1)

model = joblib.load(model_path)
print("Model loaded.")

# Define input matching the user's screenshot
input_data = {
    'credit_score': [300],
    'age': [55],
    'tenure': [5],
    'balance': [1000.0],
    'products_number': [2],
    'credit_card': [0],
    'active_member': [0],
    'estimated_salary': [10.0]
}

df = pd.DataFrame(input_data)
print("Input Data:")
print(df)

# Predict
try:
    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]
    
    print(f"\nPrediction: {pred} ({'Churn' if pred == 1 else 'No Churn'})")
    print(f"Probability: {prob:.4f}")
    
    if pred == 1:
        print("✅ Logic Correct: High Risk detected.")
    else:
        print("❌ Logic Issue: Model predicted Low Risk for High Risk input.")
except Exception as e:
    print(f"Error: {e}")
