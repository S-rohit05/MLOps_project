# End-to-End MLOps Project

This project implements a complete Machine Learning pipeline for **Customer Churn Prediction**.

## Features
- **Data Ingestion**: Synthetic data generation.
- **Model Training**: Random Forest with MLflow tracking.
- **API**: FastAPI for serving predictions.
- **Docker**: Containerized application.

## Setup

1. **Install Dependencies**
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Run Pipeline**
   ```bash
   # Generate data
   python src/components/data_ingestion.py
   
   # Train model
   python src/components/model_trainer.py
   ```

3. **Run API**
   ```bash
   uvicorn main:app --reload
   ```
   Visit `http://localhost:8000/docs` to test the API.

4. **Docker**
   ```bash
   docker build -t churn-api .
   docker run -p 8000:8000 churn-api
   ```

## Tools
- **MLflow**: `mlflow ui` to view experiments.
- **DVC**: `dvc repro` to run the pipeline.

## Quick Start (Windows)

To run the project on your PC:

1. **Activate Environment**
   ```powershell
   .\venv\Scripts\activate
   ```

2. **Run API** (in one terminal)
   ```powershell
   uvicorn main:app --reload
   ```
   Open http://localhost:8000/docs to test predictions.

3. **Run MLflow UI** (in another terminal)
   ```powershell
   mlflow ui
   ```
   Open http://localhost:5000 to view experiments.

