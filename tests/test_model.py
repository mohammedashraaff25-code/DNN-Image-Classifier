"""Unit tests for src.model."""

import numpy as np
import pytest
from tensorflow import keras
from tensorflow.keras import layers

from src.model import (
    DEFAULT_DROPOUT_RATES,
    DEFAULT_HIDDEN_UNITS,
    build_model,
    load_best_model,
)


@pytest.fixture(scope="module")
def default_model():
    return build_model()


class TestArchitecture:
    def test_default_parameter_count(self, default_model):
        assert default_model.count_params() == 577_930

    def test_input_and_output_shapes(self, default_model):
        assert default_model.input_shape == (None, 784)
        assert default_model.output_shape == (None, 10)

    def test_output_is_a_probability_distribution(self, default_model):
        batch = np.random.default_rng(0).random((8, 784)).astype("float32")
        probabilities = default_model.predict(batch, verbose=0)
        assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)
        assert (probabilities >= 0).all()

    def test_has_no_convolutional_layers(self, default_model):
        names = [layer.__class__.__name__.lower() for layer in default_model.layers]
        assert not any("conv" in name for name in names)

    def test_block_structure_matches_the_configuration(self, default_model):
        dense = [layer for layer in default_model.layers if isinstance(layer, layers.Dense)]
        batch_norms = [
            layer for layer in default_model.layers if isinstance(layer, layers.BatchNormalization)
        ]
        dropouts = [layer for layer in default_model.layers if isinstance(layer, layers.Dropout)]

        assert [layer.units for layer in dense] == [*DEFAULT_HIDDEN_UNITS, 10]
        assert len(batch_norms) == len(DEFAULT_HIDDEN_UNITS)  # one BatchNorm per hidden Dense
        assert [layer.rate for layer in dropouts] == list(DEFAULT_DROPOUT_RATES)

    def test_hidden_layers_use_he_normal_initialization(self, default_model):
        hidden = [
            layer
            for layer in default_model.layers
            if isinstance(layer, layers.Dense) and layer.units != 10
        ]
        assert all(layer.kernel_initializer.__class__.__name__ == "HeNormal" for layer in hidden)

    def test_custom_architecture(self):
        model = build_model(hidden_units=(64, 32), dropout_rates=(0.1, 0.2))
        dense_units = [layer.units for layer in model.layers if isinstance(layer, layers.Dense)]
        assert dense_units == [64, 32, 10]

    def test_custom_input_shape_and_classes(self):
        model = build_model(
            input_shape=(20,), num_classes=3, hidden_units=(8,), dropout_rates=(0.0,)
        )
        assert model.output_shape == (None, 3)


class TestCompilation:
    def test_default_loss_is_sparse_categorical_crossentropy(self, default_model):
        assert default_model.loss == "sparse_categorical_crossentropy"

    def test_loss_can_be_changed_for_one_hot_labels(self):
        model = build_model(
            hidden_units=(8,), dropout_rates=(0.1,), loss="categorical_crossentropy"
        )
        assert model.loss == "categorical_crossentropy"

    def test_model_trains_for_one_step(self):
        model = build_model(hidden_units=(16,), dropout_rates=(0.1,))
        rng = np.random.default_rng(0)
        x = rng.random((64, 784)).astype("float32")
        y = rng.integers(0, 10, size=64)
        history = model.fit(x, y, epochs=1, batch_size=32, verbose=0)
        assert np.isfinite(history.history["loss"][0])


class TestValidation:
    @pytest.mark.parametrize(
        ("units", "rates", "message"),
        [
            ((), (), "at least one"),
            ((64, 32), (0.3,), "same length"),
            ((64, 0), (0.1, 0.1), "positive"),
            ((64,), (1.0,), "dropout"),
            ((64,), (-0.1,), "dropout"),
        ],
    )
    def test_invalid_architecture_raises(self, units, rates, message):
        with pytest.raises(ValueError, match=message):
            build_model(hidden_units=units, dropout_rates=rates)


class TestLoadBestModel:
    def test_missing_file_raises_helpful_error(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="src.train"):
            load_best_model(tmp_path / "missing.keras")

    def test_save_and_load_round_trip(self, tmp_path):
        model = build_model(hidden_units=(8,), dropout_rates=(0.1,))
        path = tmp_path / "model.keras"
        model.save(path)

        restored = load_best_model(path)
        assert isinstance(restored, keras.Model)
        batch = np.random.default_rng(1).random((4, 784)).astype("float32")
        assert np.allclose(model.predict(batch, verbose=0), restored.predict(batch, verbose=0))
