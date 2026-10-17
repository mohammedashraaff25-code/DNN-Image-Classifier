"""Unit tests for src.train."""

import json

import pandas as pd
import pytest
from tensorflow import keras

from src.train import (
    TrainConfig,
    create_callbacks,
    plot_training_curves,
    save_config,
    save_history,
)


@pytest.fixture
def history():
    return {
        "loss": [1.0, 0.6, 0.4],
        "accuracy": [0.6, 0.8, 0.9],
        "val_loss": [0.9, 0.5, 0.55],
        "val_accuracy": [0.65, 0.82, 0.8],
        "lr": [1e-3, 1e-3, 5e-4],
    }


class TestTrainConfig:
    def test_defaults_match_the_documented_setup(self):
        config = TrainConfig()
        assert config.epochs == 100
        assert config.batch_size == 128
        assert config.learning_rate == pytest.approx(1e-3)
        assert config.val_split == pytest.approx(0.15)
        assert config.es_patience == 12

    def test_values_can_be_overridden(self):
        assert TrainConfig(epochs=5, batch_size=32).epochs == 5


class TestCallbacks:
    def test_the_three_overfitting_callbacks_are_configured(self, tmp_path):
        callbacks = create_callbacks(TrainConfig(), tmp_path / "models" / "best.keras")
        by_type = {type(callback): callback for callback in callbacks}

        early_stopping = by_type[keras.callbacks.EarlyStopping]
        assert early_stopping.monitor == "val_loss"
        assert early_stopping.patience == 12
        assert early_stopping.restore_best_weights is True

        reduce_lr = by_type[keras.callbacks.ReduceLROnPlateau]
        assert reduce_lr.monitor == "val_loss"
        assert reduce_lr.factor == pytest.approx(0.5)
        assert reduce_lr.patience < early_stopping.patience  # LR drops before we give up

        checkpoint = by_type[keras.callbacks.ModelCheckpoint]
        assert checkpoint.monitor == "val_loss"
        assert checkpoint.save_best_only is True

    def test_checkpoint_folder_is_created(self, tmp_path):
        target = tmp_path / "models" / "best.keras"
        create_callbacks(TrainConfig(), target)
        assert target.parent.is_dir()

    def test_patience_comes_from_the_config(self, tmp_path):
        callbacks = create_callbacks(TrainConfig(es_patience=3), tmp_path / "best.keras")
        early_stopping = next(c for c in callbacks if isinstance(c, keras.callbacks.EarlyStopping))
        assert early_stopping.patience == 3


class TestSavingArtifacts:
    def test_history_csv_has_one_row_per_epoch(self, tmp_path, history):
        path = tmp_path / "reports" / "history.csv"
        save_history(history, path)

        saved = pd.read_csv(path)
        assert list(saved["epoch"]) == [1, 2, 3]
        assert {"loss", "val_loss", "accuracy", "val_accuracy"} <= set(saved.columns)

    def test_config_json_round_trip(self, tmp_path):
        path = tmp_path / "config.json"
        save_config(TrainConfig(epochs=7, seed=1), path)

        saved = json.loads(path.read_text())
        assert saved["epochs"] == 7
        assert saved["seed"] == 1

    @pytest.mark.parametrize("with_lr", [True, False])
    def test_training_curves_are_saved(self, tmp_path, history, with_lr):
        if not with_lr:
            history.pop("lr")
        path = tmp_path / "figures" / "curves.png"

        plot_training_curves(history, path)

        assert path.exists() and path.stat().st_size > 0
