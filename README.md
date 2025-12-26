# Customer Churn Prediction MLOps

## 🌟 Overview
A production-ready **MLOps** project for predicting customer churn.
Includes:
*   **Probabilistic XGBoost Model** (Recall: ~72%)
*   **FAANG-Style Analytics Dashboard**
*   **FastAPI** Backend with Explanations
*   **CI/CD Pipeline** (check `.github/workflows`)
*   **Dockerized Deployment**

## 🚀 Quick Start
### 1. Run with Docker (Recommended)
This will start the API, MLflow, and Dashboard.
```bash
docker-compose up --build
```
Access the dashboard at: `http://localhost:8000`

### 2. Run Locally (Dev)
```bash
# Install dependencies
pip install -r requirements.txt

# Train Model
python src/components/data_ingestion.py
python src/components/model_trainer.py

# Run Server
uvicorn main:app --reload
```

## ☁️ Cloud Deployment
### CI/CD Pipeline
Every push to `main` triggers:
1.  **Dependency Check**
2.  **Model Retraining** (Reproducibility)
3.  **Performance Guardrails** (Fails if F1-Score < 0.55)

### Deploying to Render/Railway/AWS
1.  **Container**: The project is Docker-ready.
2.  **Entrypoint**: Uses `gunicorn` for production performance.
3.  **Port**: Exposes port `8000`.

## 🧪 Testing
Run the model quality gate:
```bash
pytest tests/test_model_performance.py
```

## 📊 Project Structure
*   `src/`: Application source code
*   `model/`: Saved artifacts
*   `tests/`: CI/CD guardrails
*   `.github/`: Automation workflows
