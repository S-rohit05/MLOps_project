import pytest
import numpy as np
from src.components.model_trainer import ModelTrainer


def test_metrics_calculation():
    trainer = ModelTrainer()
    y_true = [1, 0, 1, 0]
    y_pred = [1, 0, 1, 0]

    acc, prec, rec, f1 = trainer.eval_metrics(y_true, y_pred)

    assert acc == 1.0
    assert prec == 1.0
    assert rec == 1.0
    assert f1 == 1.0


def test_metrics_mismatch():
    trainer = ModelTrainer()
    y_true = [1, 0, 1, 0]
    y_pred = [0, 0, 1, 0]  # First one wrong

    acc, prec, rec, f1 = trainer.eval_metrics(y_true, y_pred)

    assert acc == 0.75
