"""Unit tests for src.evaluate."""

import json

import numpy as np
import pytest

from src.data_loader import CLASS_NAMES, NUM_CLASSES
from src.evaluate import (
    compute_classification_report,
    find_top_confusions,
    plot_confusion_matrix,
    plot_misclassified,
    plot_per_class_metrics,
    save_results,
)


@pytest.fixture
def confusion():
    cm = np.diag([100] * NUM_CLASSES)
    cm[6, 0] = 12  # Shirt predicted as T-shirt/top
    cm[0, 6] = 7  # T-shirt/top predicted as Shirt
    cm[2, 4] = 3  # Pullover predicted as Coat
    return cm


class TestTopConfusions:
    def test_pairs_are_sorted_by_count(self, confusion):
        pairs = find_top_confusions(confusion, top_k=3)
        assert [pair["count"] for pair in pairs] == [12, 7, 3]
        assert pairs[0] == {"true": "Shirt", "predicted": "T-shirt/top", "count": 12}

    def test_correct_predictions_on_the_diagonal_are_ignored(self, confusion):
        pairs = find_top_confusions(confusion, top_k=1)
        assert pairs[0]["true"] != pairs[0]["predicted"]


class TestClassificationReport:
    def test_perfect_predictions_score_one(self):
        y = np.repeat(np.arange(NUM_CLASSES), 5)
        text, report = compute_classification_report(y, y)
        assert report["accuracy"] == pytest.approx(1.0)
        assert all(report[name]["f1-score"] == pytest.approx(1.0) for name in CLASS_NAMES)
        assert "T-shirt/top" in text

    def test_wrong_predictions_lower_the_recall(self):
        y_true = np.repeat(np.arange(NUM_CLASSES), 4)
        y_pred = y_true.copy()
        y_pred[:2] = 9  # two T-shirts are predicted as Ankle boot
        _, report = compute_classification_report(y_true, y_pred)
        assert report["T-shirt/top"]["recall"] == pytest.approx(0.5)


class TestPlots:
    @pytest.mark.parametrize("normalize", [True, False])
    def test_confusion_matrix_is_saved(self, tmp_path, confusion, normalize):
        path = tmp_path / "cm.png"
        plot_confusion_matrix(confusion, path, normalize=normalize, accuracy=0.9)
        assert path.exists() and path.stat().st_size > 0

    def test_per_class_chart_is_saved(self, tmp_path):
        y = np.repeat(np.arange(NUM_CLASSES), 5)
        _, report = compute_classification_report(y, y)
        path = tmp_path / "per_class.png"
        plot_per_class_metrics(report, path)
        assert path.exists()

    @pytest.fixture
    def predictions(self):
        rng = np.random.default_rng(0)
        images = rng.random((30, 784)).astype("float32")
        y_true = rng.integers(0, NUM_CLASSES, size=30)
        y_pred = (y_true + 1) % NUM_CLASSES  # every prediction is wrong
        probabilities = np.full((30, NUM_CLASSES), 0.01)
        probabilities[np.arange(30), y_pred] = 0.91
        return images, y_true, y_pred, probabilities

    @pytest.mark.parametrize("mode", ["confident", "random"])
    def test_misclassified_grid_is_saved(self, tmp_path, predictions, mode):
        path = tmp_path / "errors.png"
        plot_misclassified(*predictions, path, n_examples=10, mode=mode)
        assert path.exists()

    def test_nothing_is_saved_when_there_are_no_errors(self, tmp_path, predictions):
        images, y_true, _, probabilities = predictions
        path = tmp_path / "errors.png"
        plot_misclassified(images, y_true, y_true, probabilities, path)
        assert not path.exists()

    def test_unknown_mode_raises(self, tmp_path, predictions):
        with pytest.raises(ValueError, match="mode"):
            plot_misclassified(*predictions, tmp_path / "x.png", mode="bogus")


def test_save_results_writes_all_report_files(tmp_path):
    y = np.repeat(np.arange(NUM_CLASSES), 5)
    text, report = compute_classification_report(y, y)

    save_results(0.25, 0.9, text, report, [], n_test=50, n_errors=5, reports_dir=tmp_path)

    metrics = json.loads((tmp_path / "evaluation_metrics.json").read_text())
    assert metrics["test_accuracy"] == pytest.approx(0.9)
    assert metrics["n_misclassified"] == 5
    assert set(metrics["per_class"]) == set(CLASS_NAMES)
    assert (tmp_path / "classification_report.txt").exists()
    assert (tmp_path / "classification_report.csv").exists()
