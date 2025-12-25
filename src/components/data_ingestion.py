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
            # Generate synthetic Customer Churn data
            logging.info("Generating synthetic Customer Churn data")
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
                "product_price": np.random.uniform(10, 10000, n_samples), # Increased max product price
            }
            df = pd.DataFrame(data)

            # --- Probabilistic Churn Logic ---
            # Instead of hard rules, we calculate a "Risk Score" (logits) and convert to probability.
            
            # Normalize features for risk calculation
            age_norm = df["age"] / 100.0
            # Affordability Ratio: Price vs Balance (Capped at 1.0)
            balance_pressure = (df["product_price"] / (df["balance"] + 1.0)).clip(upper=1.0)
            # Income Ratio: Price vs Monthly Salary (Capped at 1.0)
            salary_pressure = (df["product_price"] / ((df["estimated_salary"] / 12) + 1.0)).clip(upper=1.0)
            
            # Calculate Log-Odds (Logits)
            # Base risk bias
            logits = -2.0 
            
            # Risk Increases (+):
            logits += 3.5 * age_norm          # Older customers = Higher risk
            logits += 2.0 * balance_pressure  # Low balance relative to price = High risk
            logits += 1.5 * salary_pressure   # Low salary relative to price = High risk
            
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
