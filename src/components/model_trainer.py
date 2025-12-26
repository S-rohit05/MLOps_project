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
import pathlib

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

            logging.info("Starting model training")

            # MLflow tracking - SAFE MODE
            import tempfile
            tracking_dir = tempfile.mkdtemp()
            uri = pathlib.Path(tracking_dir).as_uri()
            mlflow.set_tracking_uri(uri)
            
            # Explicitly define artifact location to avoid permission errors
            exp_name = "CustomerChurn_CI_Fixed"
            
            try:
                mlflow.create_experiment(exp_name, artifact_location=uri)
            except mlflow.exceptions.MlflowException:
                pass

            mlflow.set_experiment(exp_name)

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

                # Calculate scale_pos_weight for imbalance handling
                # scale_pos_weight = total_neg / total_pos
                pos_count = y_train.sum()
                neg_count = len(y_train) - pos_count
                scale_weight = neg_count / pos_count if pos_count > 0 else 1.0
                
                logging.info(f"Imbalance detected. Using scale_pos_weight={scale_weight:.2f}")
                mlflow.log_param("scale_pos_weight", scale_weight)

                # Create a pipeline with scaling and model
                pipeline = Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        (
                            "model",
                            XGBClassifier(
                                n_estimators=200,          # Increased for better convergence
                                learning_rate=0.05,        # Lower LR for better generalization
                                max_depth=4,               # Reduced depth to prevent overfitting
                                scale_pos_weight=scale_weight, # Vital for recal/imbalance
                                random_state=42,
                                use_label_encoder=False,
                                eval_metric="logloss",
                            ),
                        ),
                    ]
                )

                pipeline.fit(X_train, y_train)

                # Train metrics
                train_predicted = pipeline.predict(X_train)
                tr_acc, tr_prec, tr_rec, tr_f1 = self.eval_metrics(y_train, train_predicted)
                
                # Test metrics
                test_predicted = pipeline.predict(X_test)
                accuracy, precision, recall, f1 = self.eval_metrics(y_test, test_predicted)
                
                # Calculate Probabilities for AUC
                test_proba = pipeline.predict_proba(X_test)[:, 1]
                from sklearn.metrics import roc_auc_score, average_precision_score
                roc_auc = roc_auc_score(y_test, test_proba)
                pr_auc = average_precision_score(y_test, test_proba)

                # --- Threshold Tuning ---
                # Find optimal threshold for F1
                import numpy as np
                thresholds = np.arange(0.1, 0.9, 0.05)
                best_f1 = 0
                best_thresh = 0.5
                best_recall = 0
                
                for thresh in thresholds:
                    # Convert probabilities to class labels based on threshold
                    preds = (test_proba >= thresh).astype(int)
                    f1_val = f1_score(y_test, preds)
                    if f1_val > best_f1:
                        best_f1 = f1_val
                        best_thresh = thresh
                        best_recall = recall_score(y_test, preds)

                logging.info(f"Threshold Tuning: Best Threshold={best_thresh:.2f}, Best F1={best_f1:.4f}, Recall={best_recall:.4f}")
                mlflow.log_param("optimal_threshold", best_thresh)
                mlflow.log_metric("optimized_f1", best_f1)
                mlflow.log_metric("optimized_recall", best_recall)

                logging.info(
                    f"Default Metrics (0.5): Recall={recall:.4f}, F1={f1:.4f}, PR-AUC={pr_auc:.4f} (Accuracy={accuracy:.4f})"
                )

                # Log training metrics
                mlflow.log_metric("train_recall", tr_rec)
                mlflow.log_metric("train_f1", tr_f1)
                mlflow.log_metric("train_precision", tr_prec)

                # Log test metrics
                mlflow.log_metric("test_recall", recall)
                mlflow.log_metric("test_f1", f1)
                mlflow.log_metric("test_precision", precision)
                mlflow.log_metric("test_roc_auc", roc_auc)
                mlflow.log_metric("test_pr_auc", pr_auc)

                # Log params
                mlflow.log_param("n_estimators", 200)
                mlflow.log_param("learning_rate", 0.05)
                mlflow.log_param("max_depth", 4)

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
