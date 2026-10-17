"""Data loading and preprocessing utilities for Fashion-MNIST.

This module prepares Fashion-MNIST for a fully connected (dense) neural
network. The full pipeline performed by :func:`load_data` is:

1. Load the raw dataset from ``tensorflow.keras.datasets``.
2. Create a stratified validation split from the training set.
3. Flatten each image from ``(28, 28)`` to a ``(784,)`` vector.
4. Normalize pixel values from ``[0, 255]`` to ``[0.0, 1.0]``.
5. Optionally one-hot encode the labels.

Each step is also exposed as an independent function so it can be reused and
tested on its own.

Example:
    >>> from src.data_loader import load_data
    >>> X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    >>> X_train.shape
    (51000, 784)

Run a self-test from the project root with::

    python -m src.data_loader
"""

from __future__ import annotations

import logging

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
NUM_CLASSES: int = 10
IMG_HEIGHT: int = 28
IMG_WIDTH: int = 28
INPUT_DIM: int = IMG_HEIGHT * IMG_WIDTH  # 784 features per flattened image
MAX_PIXEL_VALUE: float = 255.0

DEFAULT_VAL_SPLIT: float = 0.15
DEFAULT_SEED: int = 42

CLASS_NAMES: list[str] = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

# Type alias: (X_train, y_train, X_val, y_val, X_test, y_test)
DataSplits = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


# --------------------------------------------------------------------------- #
# Individual pipeline steps
# --------------------------------------------------------------------------- #
def load_raw_data() -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Load the raw Fashion-MNIST dataset.

    The dataset is downloaded automatically on first use and cached in
    ``~/.keras/datasets/``.

    Returns:
        A tuple ``((x_train, y_train), (x_test, y_test))`` where images are
        ``uint8`` arrays of shape ``(N, 28, 28)`` and labels are ``uint8``
        arrays of shape ``(N,)`` with values in ``[0, 9]``.
    """
    logger.info("Loading Fashion-MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    logger.info("Loaded raw data: train=%s, test=%s", x_train.shape, x_test.shape)
    return (x_train, y_train), (x_test, y_test)


def flatten_images(images: np.ndarray) -> np.ndarray:
    """Flatten images from ``(N, 28, 28)`` to ``(N, 784)``.

    Args:
        images: Array of shape ``(N, height, width)``.

    Returns:
        Array of shape ``(N, height * width)``.

    Raises:
        ValueError: If ``images`` is not a 3-dimensional array.
    """
    if images.ndim != 3:
        raise ValueError(f"Expected images with shape (N, H, W), got {images.shape}.")
    return images.reshape(images.shape[0], -1)


def normalize_images(images: np.ndarray, dtype: type = np.float32) -> np.ndarray:
    """Scale pixel values from ``[0, 255]`` to ``[0.0, 1.0]``.

    Args:
        images: Array of raw pixel values (typically ``uint8``).
        dtype: Floating point dtype of the returned array.

    Returns:
        Array with the same shape as ``images`` and values in ``[0, 1]``.
    """
    return images.astype(dtype) / MAX_PIXEL_VALUE


def stratified_train_val_split(
    x: np.ndarray,
    y: np.ndarray,
    val_split: float = DEFAULT_VAL_SPLIT,
    seed: int = DEFAULT_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data into train and validation sets, preserving class proportions.

    The split is stratified: every class contributes ``val_split`` of its
    samples to the validation set, so class balance is identical in both
    subsets. The result is deterministic for a given ``seed``.

    Args:
        x: Feature array of shape ``(N, ...)``.
        y: Integer label array of shape ``(N,)``.
        val_split: Fraction of samples to use for validation, in ``(0, 1)``.
        seed: Random seed for reproducible shuffling.

    Returns:
        A tuple ``(x_train, y_train, x_val, y_val)``.

    Raises:
        ValueError: If ``val_split`` is not in ``(0, 1)`` or if ``x`` and
            ``y`` have different lengths.
    """
    if not 0.0 < val_split < 1.0:
        raise ValueError(f"val_split must be in (0, 1), got {val_split}.")
    if len(x) != len(y):
        raise ValueError(f"x and y must have the same length, got {len(x)} and {len(y)}.")

    rng = np.random.default_rng(seed)

    val_indices = []
    for class_id in np.unique(y):
        class_indices = np.flatnonzero(y == class_id)
        rng.shuffle(class_indices)
        n_val = int(round(len(class_indices) * val_split))
        val_indices.append(class_indices[:n_val])
    val_idx = np.concatenate(val_indices)

    train_mask = np.ones(len(y), dtype=bool)
    train_mask[val_idx] = False
    train_idx = np.flatnonzero(train_mask)

    # Shuffle so samples are not grouped by class.
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def one_hot_encode(labels: np.ndarray, num_classes: int = NUM_CLASSES) -> np.ndarray:
    """Convert integer class labels to one-hot vectors.

    Args:
        labels: Integer label array of shape ``(N,)``.
        num_classes: Total number of classes.

    Returns:
        ``float32`` array of shape ``(N, num_classes)``.

    Raises:
        ValueError: If ``labels`` is not 1-dimensional.
    """
    if labels.ndim != 1:
        raise ValueError(
            f"Expected labels with shape (N,), got {labels.shape}. "
            "Labels may already be one-hot encoded."
        )
    return tf.keras.utils.to_categorical(labels, num_classes=num_classes).astype(np.float32)


# --------------------------------------------------------------------------- #
# Full pipeline
# --------------------------------------------------------------------------- #
def load_data(
    val_split: float = DEFAULT_VAL_SPLIT,
    one_hot: bool = False,
    flatten: bool = True,
    normalize: bool = True,
    seed: int = DEFAULT_SEED,
) -> DataSplits:
    """Load Fashion-MNIST and return preprocessed train/val/test splits.

    Args:
        val_split: Fraction of the training set held out for validation
            (default ``0.15``, i.e. 51,000 train / 9,000 validation).
        one_hot: If ``True``, labels are one-hot encoded with shape
            ``(N, 10)``. If ``False``, labels stay as integers with shape
            ``(N,)``. Use one-hot with ``categorical_crossentropy`` and
            integer labels with ``sparse_categorical_crossentropy``.
        flatten: If ``True``, images are flattened to ``(N, 784)``.
        normalize: If ``True``, pixels are scaled to ``[0, 1]`` as ``float32``.
        seed: Random seed used for the validation split.

    Returns:
        A tuple ``(X_train, y_train, X_val, y_val, X_test, y_test)``.
    """
    (x_train_full, y_train_full), (x_test, y_test) = load_raw_data()

    # Split on raw data first. Normalization is a fixed /255 (no statistics
    # learned from the data), so there is no leakage either way.
    x_train, y_train, x_val, y_val = stratified_train_val_split(
        x_train_full, y_train_full, val_split=val_split, seed=seed
    )

    if flatten:
        x_train, x_val, x_test = (
            flatten_images(x_train),
            flatten_images(x_val),
            flatten_images(x_test),
        )

    if normalize:
        x_train, x_val, x_test = (
            normalize_images(x_train),
            normalize_images(x_val),
            normalize_images(x_test),
        )

    if one_hot:
        y_train, y_val, y_test = (
            one_hot_encode(y_train),
            one_hot_encode(y_val),
            one_hot_encode(y_test),
        )

    logger.info(
        "Prepared splits: train=%s, val=%s, test=%s",
        x_train.shape,
        x_val.shape,
        x_test.shape,
    )
    return x_train, y_train, x_val, y_val, x_test, y_test


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
def _describe(name: str, x: np.ndarray, y: np.ndarray) -> None:
    """Print a short summary of one data split."""
    print(
        f"{name:<6} X: {str(x.shape):<14} {x.dtype!s:<8} "
        f"range=[{x.min():.2f}, {x.max():.2f}] | "
        f"y: {str(y.shape):<12} {y.dtype}"
    )


def _class_counts(y: np.ndarray) -> np.ndarray:
    """Return per-class sample counts for integer or one-hot labels."""
    labels = y.argmax(axis=1) if y.ndim == 2 else y
    return np.bincount(labels, minlength=NUM_CLASSES)


def main() -> None:
    """Run a quick sanity check of the full data pipeline."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    # --- Default: integer labels ------------------------------------------- #
    print("\n=== Integer labels ===")
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(one_hot=False)
    _describe("train", X_train, y_train)
    _describe("val", X_val, y_val)
    _describe("test", X_test, y_test)

    assert X_train.shape == (51_000, INPUT_DIM)
    assert X_val.shape == (9_000, INPUT_DIM)
    assert X_test.shape == (10_000, INPUT_DIM)
    assert X_train.dtype == np.float32
    assert X_train.min() >= 0.0 and X_train.max() <= 1.0
    assert len(X_train) + len(X_val) == 60_000

    # Stratification check: classes should be equally represented.
    print("\nSamples per class:")
    print(f"  train: {_class_counts(y_train).tolist()}")
    print(f"  val  : {_class_counts(y_val).tolist()}")
    print(f"  test : {_class_counts(y_test).tolist()}")
    assert (_class_counts(y_val) == 900).all()

    # --- One-hot labels ----------------------------------------------------- #
    print("\n=== One-hot labels ===")
    _, y_train_oh, _, y_val_oh, _, y_test_oh = load_data(one_hot=True)
    _describe("train", X_train, y_train_oh)
    _describe("val", X_val, y_val_oh)
    _describe("test", X_test, y_test_oh)

    assert y_train_oh.shape == (51_000, NUM_CLASSES)
    assert (y_train_oh.sum(axis=1) == 1).all()
    assert np.array_equal(y_train_oh.argmax(axis=1), y_train)

    # --- Reproducibility ---------------------------------------------------- #
    again = load_data(one_hot=False)
    assert np.array_equal(y_train, again[1]), "Split is not reproducible!"

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
