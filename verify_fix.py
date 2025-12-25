import joblib
import pandas as pd
import os

def check_rich_user_prediction():
    model = joblib.load(os.path.join("model", "model.joblib"))
    
    # Case: Millionaire buying a $5k product
    rich_user = pd.DataFrame([{
        "credit_score": 800,
        "age": 35,
        "tenure": 5,
        "balance": 500000.0,
        "products_number": 2,
        "credit_card": 1,
        "active_member": 1,
        "estimated_salary": 1000000.0,
        "product_price": 5000.0
    }])
    
    pred = model.predict(rich_user)[0]
    prob = model.predict_proba(rich_user)[0][1]
    
    print(f"\n--- Rich User Prediction ---")
    print(f"Salary: $1M, Price: $5k")
    print(f"Prediction: {'Churn 🔴' if pred==1 else 'Safe 🟢'}")
    print(f"Churn Probability: {prob:.2%}")

if __name__ == "__main__":
    check_rich_user_prediction()
