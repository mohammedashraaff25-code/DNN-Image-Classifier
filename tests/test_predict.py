"""Unit tests for src.predict."""

import numpy as np
import pytest
from PIL import Image, ImageDraw

from src.data_loader import CLASS_NAMES
from src.model import build_model
from src.predict import (
    load_image,
    plot_predictions,
    predict_image,
    predict_probabilities,
    top_k_predictions,
)


def _make_photo(path, background, item, size=(120, 180)):
    """Create a non-square test photo: a rectangular 'item' on a plain background."""
    image = Image.new("L", size, background)
    ImageDraw.Draw(image).rectangle([30, 30, 90, 150], fill=item)
    image.save(path)
    return path


class TestLoadImage:
    def test_output_format(self, tmp_path):
        path = _make_photo(tmp_path / "photo.png", background=255, item=30)
        image = load_image(path)
        assert image.shape == (28, 28)
        assert image.dtype == np.float32
        assert image.min() >= 0.0 and image.max() <= 1.0

    def test_bright_background_is_inverted_automatically(self, tmp_path):
        """A dark item on a white background becomes a light item on black (MNIST style)."""
        path = _make_photo(tmp_path / "photo.png", background=255, item=30)
        image = load_image(path, invert="auto")
        assert image[0, 0] < 0.1  # background is now dark
        assert image[14, 14] > 0.7  # item is now light

    def test_dark_background_is_left_alone(self, tmp_path):
        path = _make_photo(tmp_path / "photo.png", background=0, item=220)
        image = load_image(path, invert="auto")
        assert image[0, 0] < 0.1
        assert image[14, 14] > 0.7

    def test_manual_invert_overrides_auto_detection(self, tmp_path):
        path = _make_photo(tmp_path / "photo.png", background=255, item=30)
        assert load_image(path, invert="no")[0, 0] > 0.9
        assert load_image(path, invert="yes")[0, 0] < 0.1

    def test_invalid_invert_option_raises(self, tmp_path):
        path = _make_photo(tmp_path / "photo.png", background=255, item=30)
        with pytest.raises(ValueError, match="invert"):
            load_image(path, invert="maybe")

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_image(tmp_path / "missing.jpg")


class TestPredictions:
    def test_top_k_is_sorted_and_uses_class_names(self):
        probabilities = np.zeros(10)
        probabilities[[3, 7, 1]] = [0.5, 0.3, 0.2]
        top = top_k_predictions(probabilities, k=3)
        assert [name for name, _ in top] == [CLASS_NAMES[3], CLASS_NAMES[7], CLASS_NAMES[1]]
        assert [p for _, p in top] == pytest.approx([0.5, 0.3, 0.2])

    @pytest.fixture(scope="class")
    def small_model(self):
        return build_model(hidden_units=(16,), dropout_rates=(0.1,))

    def test_probabilities_accept_image_or_flat_input(self, small_model):
        images = np.random.default_rng(0).random((3, 28, 28)).astype("float32")
        from_images = predict_probabilities(small_model, images)
        from_flat = predict_probabilities(small_model, images.reshape(3, -1))
        assert from_images.shape == (3, 10)
        assert np.allclose(from_images, from_flat)

    def test_predict_image_returns_k_sorted_predictions(self, small_model, tmp_path):
        path = _make_photo(tmp_path / "photo.png", background=255, item=30)
        predictions = predict_image(small_model, path, top_k=3)
        assert len(predictions) == 3
        scores = [score for _, score in predictions]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.parametrize("with_truth", [True, False])
    def test_prediction_grid_is_saved(self, small_model, tmp_path, with_truth):
        images = np.random.default_rng(0).random((6, 28, 28)).astype("float32")
        probabilities = predict_probabilities(small_model, images)
        path = tmp_path / "grid.png"

        plot_predictions(
            images,
            probabilities,
            path,
            true_labels=np.arange(6) if with_truth else None,
            names=None if with_truth else [f"img{i}.png" for i in range(6)],
        )

        assert path.exists() and path.stat().st_size > 0
