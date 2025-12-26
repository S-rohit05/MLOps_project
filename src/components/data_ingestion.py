import os
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
import logging

# Basic logging setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("data", "raw.csv")
    train_data_path: str = os.path.join("data", "train.csv")
    test_data_path: str = os.path.join("data", "test.csv")


class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            external_data_path = os.path.join("data", "external_data.csv")
            
            if os.path.exists(external_data_path):
                # --- Load External Data ---
                logging.info(f"External dataset found at {external_data_path}. Loading...")
                df = pd.read_csv(external_data_path)
                
                # Column Mapping (Improvising based on standard datasets)
                column_mapping = {
                    'CreditScore': 'credit_score',
                    'Age': 'age',
                    'Tenure': 'tenure',
                    'Balance': 'balance',
                    'NumOfProducts': 'products_number',
                    'HasCrCard': 'credit_card',
                    'IsActiveMember': 'active_member',
                    'EstimatedSalary': 'estimated_salary',
                    'Exited': 'churn'
                }
                df = df.rename(columns=column_mapping)
                
                # Verify required columns align with our training schema
                required_cols = ['credit_score', 'age', 'tenure', 'balance', 
                                 'products_number', 'credit_card', 'active_member', 
                                 'estimated_salary', 'churn']
                
                if not all(col in df.columns for col in required_cols):
                    missing = [col for col in required_cols if col not in df.columns]
                    raise ValueError(f"External data missing required columns: {missing}")

                # Keep only relevant columns
                df = df[required_cols]
                
                # --- Feature Engineering ---
                logging.info("Applying feature engineering...")
                
                # 1. BalanceSalaryRatio
                df['balance_salary_ratio'] = df['balance'] / df['estimated_salary']
                
                # 2. TenureAgeRatio
                df['tenure_age_ratio'] = df['tenure'] / df['age']
                
                # 3. ProductsPerYear (Avoid division by zero)
                df['tenure_safe'] = df['tenure'].replace(0, 1)
                df['products_per_year'] = df['products_number'] / df['tenure_safe']
                df.drop(columns=['tenure_safe'], inplace=True)
                
                # 4. CreditCard Active Interaction
                df['is_active_cr_card'] = df['active_member'] * df['credit_card']

                logging.info(f"Loaded {len(df)} rows from external dataset.")

            else:
                # --- Fallback to Synthetic Data Generation ---
                logging.info("External data not found. Generating synthetic Customer Churn data")
                np.random.seed(42)
                n_samples = 2000  # Increased samples for better coverage

                data = {
                    "credit_score": np.random.randint(300, 850, n_samples),
                    "age": np.random.randint(18, 90, n_samples),
                    "tenure": np.random.randint(0, 10, n_samples),
                    "balance": np.random.uniform(0, 2500000, n_samples), # Increased max balance to 2.5M
                    "products_number": np.random.randint(1, 5, n_samples),
                    "credit_card": np.random.randint(0, 2, n_samples),
                    "active_member": np.random.randint(0, 2, n_samples),
                    "estimated_salary": np.random.uniform(10000, 2000000, n_samples), # Increased max salary to 2M
                }
                df = pd.DataFrame(data)

                # --- Probabilistic Churn Logic ---
                # Instead of hard rules, we calculate a "Risk Score" (logits) and convert to probability.
                
                # Normalize features for risk calculation
                age_norm = df["age"] / 100.0
                
                # Calculate Log-Odds (Logits)
                # Base risk bias
                logits = -2.0 
                
                # Risk Increases (+):
                logits += 3.5 * age_norm          # Older customers = Higher risk
                # Removed price-based pressure metrics
                
                # Risk Decreases (-):
                logits -= 1.5 * df["active_member"]       # Active members stay
                logits -= 0.1 * df["tenure"]              # Loyal customers stay
                logits -= 0.5 * df["credit_card"]         # Credit card creates slight lock-in
                logits -= 0.5 * (df["products_number"] == 2).astype(int) # 2 products is often stable
                
                # Convert Logits to Probability (Sigmoid)
                churn_prob = 1 / (1 + np.exp(-logits))
                
                # Generate Churn Label using Bernoulli Sampling
                # This allows a "High Risk" user to sometimes stay (Realistic)
                df["churn"] = np.random.binomial(n=1, p=churn_prob)
            
            # Save raw data
            os.makedirs(os.path.dirname(self.config.raw_data_path), exist_ok=True)
            df.to_csv(self.config.raw_data_path, index=False)
            logging.info(f"Raw data saved at {self.config.raw_data_path}")

            # Split data
            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.config.train_data_path, index=False, header=True)
            test_set.to_csv(self.config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return (self.config.train_data_path, self.config.test_data_path)
        except Exception as e:
            logging.error(f"Error in data ingestion: {str(e)}")
            raise e


if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()
