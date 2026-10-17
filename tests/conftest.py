"""Shared pytest fixtures.

Tests never download Fashion-MNIST: the Keras loader is replaced by a small,
balanced, random dataset, so the whole suite runs offline in seconds.
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # quiet TensorFlow logs

import matplotlib

matplotlib.use("Agg")  # headless backend for plotting tests

import numpy as np
import pytest

from src import data_loader

N_TRAIN_PER_CLASS = 100  # 1,000 training images (val split 15% -> 850 / 150)
N_TEST_PER_CLASS = 20  # 200 test images


def _random_split(rng: np.random.Generator, n_per_class: int) -> tuple[np.ndarray, np.ndarray]:
    """Create a balanced random split of ``(images, labels)``."""
    labels = np.repeat(np.arange(10, dtype="uint8"), n_per_class)
    rng.shuffle(labels)
    images = rng.integers(0, 256, size=(len(labels), 28, 28), dtype="uint8")
    return images, labels


@pytest.fixture
def fake_dataset(monkeypatch):
    """Replace ``fashion_mnist.load_data`` with a tiny offline dataset."""
    rng = np.random.default_rng(0)
    train = _random_split(rng, N_TRAIN_PER_CLASS)
    test = _random_split(rng, N_TEST_PER_CLASS)
    monkeypatch.setattr(data_loader.fashion_mnist, "load_data", lambda: (train, test))
    return train, test
