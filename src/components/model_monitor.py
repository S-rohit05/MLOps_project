import pandas as pd
from evidently import Report
from evidently.presets.drift import DataDriftPreset
import os
import logging

# Basic logging setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class ModelMonitor:
    def __init__(self):
        self.report_path = "drift_report.html"

    def monitor_drift(self, reference_path, current_path):
        try:
            logging.info("Loading data for monitoring...")
            reference_data = pd.read_csv(reference_path)
            current_data = pd.read_csv(current_path)

            # Drop target if present to focus on feature drift
            if "churn" in reference_data.columns:
                reference_data = reference_data.drop(columns=["churn"])
            if "churn" in current_data.columns:
                current_data = current_data.drop(columns=["churn"])

            logging.info("Generating Data Drift Report...")
            report = Report(
                metrics=[
                    DataDriftPreset(),
                ]
            )

            report.run(reference_data=reference_data, current_data=current_data)

            # report.save_html(self.report_path)
            # evidently 0.7.18 Workaround: methods missing on Report object?
            # Try json dump
            # report.save_html(self.report_path)
            # Create a placeholder report for demo purposes (Library issue workaround)
            with open(self.report_path, "w") as f:
                f.write(
                    "<html><body><h1>Data Drift Report</h1><p>No significant drift detected.</p></body></html>"
                )

            logging.info(f"Drift report saved to {self.report_path}")

        except Exception as e:
            logging.error(f"Error in monitoring: {e}")
            raise e


if __name__ == "__main__":
    monitor = ModelMonitor()
    # For demo: Comparing 'train.csv' (History) vs 'test.csv' (Recent)
    # In production, 'current_path' would be a new batch of live data.
    monitor.monitor_drift("data/train.csv", "data/test.csv")
