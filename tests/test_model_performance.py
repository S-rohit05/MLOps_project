import pandas as pd
import joblib
import os
import pytest
from sklearn.metrics import f1_score, recall_score

def test_model_performance():
    # Paths
    model_path = os.path.join("model", "model.joblib")
    test_data_path = os.path.join("data", "test.csv")
    
    # Ensure artifacts exist
    if not os.path.exists(model_path) or not os.path.exists(test_data_path):
        pytest.skip("Model or test data not found. Skipping performance test.")
        
    # Load resources
    pipeline = joblib.load(model_path)
    test_df = pd.read_csv(test_data_path)
    
    X_test = test_df.drop(columns=["churn"])
    y_test = test_df["churn"]
    
    # Predict (using optimal logic or default)
    # Note: Ideally we use the 'optimal_threshold' found during training, 
    # but for simplicity in this guardrail, we check the robust default performance 
    # or a known safe threshold.
    
    preds = pipeline.predict(X_test)
    
    # Metrics
    f1 = f1_score(y_test, preds)
    recall = recall_score(y_test, preds)
    
    print(f"\nGuardrail Metrics: F1={f1:.4f}, Recall={recall:.4f}")
    
    # Assertions (Baseline Guardrails)
    # The user mentioned original F1 was ~0.55. We want to ensure we stay above that.
    assert f1 > 0.55, f"Model F1 score {f1:.4f} is below acceptance criteria (0.55)"
    assert recall > 0.60, f"Model Recall {recall:.4f} is too low (target > 0.60)"

if __name__ == "__main__":
    test_model_performance()
