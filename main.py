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
        input_df = pd.DataFrame([data.dict()])
        
        # Make prediction
        prediction = model.predict(input_df)
        prob = model.predict_proba(input_df)
        
        churn_prob = prob[0][1] # Probability of Churn (class 1)
        
            "label": "Churn" if prediction[0] == 1 else "No Churn"
        }
        
        # Save to database
        save_prediction(data.dict(), int(prediction[0]), float(churn_prob))
        
        return {
            "prediction": int(prediction[0]),
            "churn_probability": float(churn_prob),
            "label": "Churn" if prediction[0] == 1 else "No Churn"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
def history():
    return get_recent_predictions()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
