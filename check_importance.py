import joblib
import pandas as pd
import os

def get_feature_importance():
    model_path = os.path.join("model", "model.joblib")
    if not os.path.exists(model_path):
        print("Model file not found.")
        return

    pipeline = joblib.load(model_path)
    
    # Extract feature names from training data to be sure
    # But since we know them:
    features = [
        'credit_score', 'age', 'tenure', 'balance', 
        'products_number', 'credit_card', 'active_member', 'estimated_salary',
        'product_price'
    ]
    
    try:
        # Access the XGB model within the pipeline
        xgb_model = pipeline.named_steps['model']
        importance = xgb_model.feature_importances_
        
        feat_imp = pd.Series(importance, index=features).sort_values(ascending=False)
        print("\n--- Feature Importance ---")
        print(feat_imp)
    except Exception as e:
        print(f"Error extracting importance: {e}")

if __name__ == "__main__":
    get_feature_importance()
