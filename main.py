from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import uvicorn
import os
from src.database import init_db, save_prediction, get_recent_predictions

app = FastAPI(
    title="Customer Churn Prediction API",
    description="API to predict if a customer will churn based on their metrics.",
    version="1.0.0"
)

# Load Model
MODEL_PATH = os.path.join("model", "model.joblib")
model = None

class CustomerData(BaseModel):
    credit_score: int
    age: int
    tenure: int
    balance: float
    products_number: int
    credit_card: int
    active_member: int
    estimated_salary: float

@app.on_event("startup")
def load_model():
    global model
    init_db()
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            print("Model loaded successfully.")
        else:
            print(f"Warning: Model not found at {MODEL_PATH}. API will not work until model is trained.")
    except Exception as e:
        print(f"Error loading model: {e}")

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse

# Mount static files
app.mount("/static", StaticFiles(directory="src/static"), name="static")

@app.get("/")
def home():
    return FileResponse("src/static/index.html")

@app.get("/monitoring", response_class=HTMLResponse)
def monitoring():
    report_path = "drift_report.html"
    if os.path.exists(report_path):
        with open(report_path, "r", encoding='utf-8') as f:
            return f.read()
    return "<h1>Drift Report not found. Run 'src/components/model_monitor.py' first.</h1>"

@app.post("/predict")
def predict(data: CustomerData):
    if not model:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    
    try:
        # Convert input to dataframe expected by the pipeline
        input_data = data.dict()
    
        # Create DataFrame for prediction
        df = pd.DataFrame([input_data])
        
        # --- Feature Engineering (Must match training logic) ---
        df['balance_salary_ratio'] = df['balance'] / df['estimated_salary']
        df['tenure_age_ratio'] = df['tenure'] / df['age']
        
        # Safe division for products_per_year
        tenure_safe = df['tenure'].replace(0, 1)
        df['products_per_year'] = df['products_number'] / tenure_safe
        
        df['is_active_cr_card'] = df['active_member'] * df['credit_card']
        
        # Make prediction
        prediction = model.predict(df)[0]
        churn_prob = model.predict_proba(df)[0][1]
        
        # Save to database
        save_prediction(data.dict(), int(prediction), float(churn_prob))
        
        # --- Explainability Logic (Simple Heuristics) ---
        factors = []
        if df['age'].iloc[0] > 50:
            factors.append("Senior Customer (>50)")
        if df['active_member'].iloc[0] == 0:
            factors.append("Inactive Member")
        if df['balance'].iloc[0] == 0:
            factors.append("Zero Balance")
        if df['products_number'].iloc[0] >= 3:
            factors.append("High Product Volatility")
        if df['credit_score'].iloc[0] < 500:
            factors.append("Low Credit Score")
        
        if not factors and churn_prob > 0.5:
            factors.append("Complex Risk Pattern")
        elif not factors and churn_prob <= 0.5:
             factors.append("Strong Loyalty Indicators")

        return {
            "prediction": int(prediction),
            "churn_probability": float(churn_prob),
            "label": "Churn" if prediction == 1 else "No Churn",
            "factors": factors
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
def history():
    return get_recent_predictions()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
