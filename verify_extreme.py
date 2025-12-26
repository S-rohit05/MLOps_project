import joblib
import pandas as pd
import os

def check_extreme_scenario():
    model = joblib.load(os.path.join("model", "model.joblib"))
    
    # Extract features from user image
    user_data = pd.DataFrame([{
        "credit_score": 650,
        "age": 35,
        "tenure": 5,
        "balance": 50.0,
        "products_number": 2,
        "credit_card": 1,
        "active_member": 1,
        "estimated_salary": 100.0,
        "product_price": 500.0
    }])
    
    pred = model.predict(user_data)[0]
    prob = model.predict_proba(user_data)[0][1]
    
    print(f"\n--- Extreme Scenario Prediction ---")
    print(f"Inputs: Balance=$50, Salary=$100, Price=$500")
    print(f"Prediction: {'Churn 🔴' if pred==1 else 'Safe 🟢'}")
    print(f"Churn Probability: {prob:.2%}")

if __name__ == "__main__":
    check_extreme_scenario()
