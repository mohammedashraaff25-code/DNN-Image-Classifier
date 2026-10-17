"""End-to-end smoke test: train -> evaluate -> predict on a tiny offline dataset."""

import json

import numpy as np

from src.evaluate import run_evaluation
from src.model import load_best_model
from src.predict import predict_probabilities
from src.train import TrainConfig, run_training


def test_train_then_evaluate_then_predict(fake_dataset, tmp_path):
    # --- train -------------------------------------------------------------- #
    config = TrainConfig(epochs=2, batch_size=64, es_patience=2)
    history = run_training(config, output_root=tmp_path)

    model_path = tmp_path / "models" / "best_model.keras"
    assert len(history["loss"]) >= 1
    assert model_path.exists()
    assert (tmp_path / "reports" / "training_history.csv").exists()
    assert (tmp_path / "reports" / "training_config.json").exists()
    assert (tmp_path / "reports" / "figures" / "07_training_curves.png").exists()

    # --- evaluate ----------------------------------------------------------- #
    reports_dir = tmp_path / "reports"
    results = run_evaluation(model_path=model_path, reports_dir=reports_dir, n_misclassified=5)

    assert 0.0 <= results["test_accuracy"] <= 1.0
    metrics = json.loads((reports_dir / "evaluation_metrics.json").read_text())
    assert metrics["n_test_samples"] == 200
    figures = reports_dir / "figures"
    assert (figures / "08_confusion_matrix_counts.png").exists()
    assert (figures / "09_confusion_matrix_normalized.png").exists()
    assert (figures / "10_per_class_metrics.png").exists()

    # --- predict ------------------------------------------------------------ #
    model = load_best_model(model_path)
    images = np.random.default_rng(0).random((4, 784)).astype("float32")
    probabilities = predict_probabilities(model, images)
    assert probabilities.shape == (4, 10)
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)
