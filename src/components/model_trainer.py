import os
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import mlflow
import mlflow.sklearn
import logging
from dataclasses import dataclass
import sys

# Basic logging setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


@dataclass
class ModelTrainerConfig:
    train_data_path: str = os.path.join("data", "train.csv")
    test_data_path: str = os.path.join("data", "test.csv")
    model_path: str = os.path.join("model", "model.joblib")


class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()
        os.makedirs(os.path.dirname(self.config.model_path), exist_ok=True)

    def eval_metrics(self, actual, pred):
        accuracy = accuracy_score(actual, pred)
        precision = precision_score(actual, pred)
        recall = recall_score(actual, pred)
        f1 = f1_score(actual, pred)
        return accuracy, precision, recall, f1

    def initiate_model_trainer(self):
        try:
            logging.info("Loading training and testing data")
            train_df = pd.read_csv(self.config.train_data_path)
            test_df = pd.read_csv(self.config.test_data_path)

            logging.info("Splitting training and testing input data")
            X_train = train_df.drop(columns=["churn"])
            y_train = train_df["churn"]
            X_test = test_df.drop(columns=["churn"])
            y_test = test_df["churn"]

            # Create a pipeline with scaling and model
            pipeline = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        XGBClassifier(
                            n_estimators=100,
                            learning_rate=0.1,
                            max_depth=5,
                            random_state=42,
                            use_label_encoder=False,
                            eval_metric="logloss",
                        ),
                    ),
                ]
            )

            logging.info("Starting model training")

            # MLflow tracking
            mlflow.set_tracking_uri("file:./mlruns_xgboost")
            mlflow.set_experiment("CustomerChurnPrediction")

            with mlflow.start_run():
                # Log tags and sample info
                mlflow.set_tag("model_type", "xgboost")
                mlflow.set_tag("stage", "evaluation")
                
                num_train = len(y_train)
                num_test = len(y_test)
                mlflow.log_param("train_samples", num_train)
                mlflow.log_param("test_samples", num_test)

                # Log class distributions
                train_churn_rate = y_train.mean()
                test_churn_rate = y_test.mean()
                mlflow.log_param("train_churn_rate", f"{train_churn_rate:.2%}")
                mlflow.log_param("test_churn_rate", f"{test_churn_rate:.2%}")
                
                logging.info(f"Class Distribution: Train Churn={train_churn_rate:.2%}, Test Churn={test_churn_rate:.2%}")

                pipeline.fit(X_train, y_train)

                # Train metrics
                train_predicted = pipeline.predict(X_train)
                tr_acc, tr_prec, tr_rec, tr_f1 = self.eval_metrics(y_train, train_predicted)
                
                # Test metrics
                test_predicted = pipeline.predict(X_test)
                accuracy, precision, recall, f1 = self.eval_metrics(y_test, test_predicted)

                logging.info(
                    f"Test Metrics: Accuracy={accuracy}, Precision={precision}, Recall={recall}, F1={f1}"
                )

                # Log training metrics
                mlflow.log_metric("train_accuracy", tr_acc)
                mlflow.log_metric("train_precision", tr_prec)
                mlflow.log_metric("train_recall", tr_rec)
                mlflow.log_metric("train_f1", tr_f1)

                # Log test metrics
                mlflow.log_metric("test_accuracy", accuracy)
                mlflow.log_metric("test_precision", precision)
                mlflow.log_metric("test_recall", recall)
                mlflow.log_metric("test_f1", f1)

                # Log params (example)
                mlflow.log_param("n_estimators", 100)
                mlflow.log_param("learning_rate", 0.1)

                # Log model
                mlflow.sklearn.log_model(pipeline, "model")

                # Save model locally
                joblib.dump(pipeline, self.config.model_path)
                logging.info(f"Model saved to {self.config.model_path}")

        except Exception as e:
            logging.error(f"Error in model training: {str(e)}")
            raise e


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.initiate_model_trainer()
