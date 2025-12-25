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

            # Create correlated labels (Churn Logic)
            df["churn"] = 0

            # 1. Senior Risk (Age > 60)
            mask_senior = (df["age"] > 60)
            
            # 2. Inactive Small Balance Risk
            mask_inactive = (df["balance"] < 10000) & (df["active_member"] == 0)
            
            # 3. New: Affordability Risk (Price vs Monthly Salary)
            # Monthly Salary = estimated_salary / 12. Risk if price > 30% of monthly income.
            mask_unaffordable_salary = (df["product_price"] > (df["estimated_salary"] / 12) * 0.3)
            
            # 4. New: Affordability Risk (Price vs Balance)
            # Risk if price > 20% of current balance.
            mask_unaffordable_balance = (df["product_price"] > df["balance"] * 0.2) & (df["balance"] > 0)

            df.loc[mask_senior | mask_inactive | mask_unaffordable_salary | mask_unaffordable_balance, "churn"] = 1

            # Add small noise (flip 2% of labels)
            flip_indices = np.random.choice(
                df.index, size=int(0.02 * len(df)), replace=False
            )
            df.loc[flip_indices, "churn"] = 1 - df.loc[flip_indices, "churn"]

            # Drop the helper col if it exists (it doesn't in this new logic)
            # df = df.drop(columns=['churn_prob'])

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
