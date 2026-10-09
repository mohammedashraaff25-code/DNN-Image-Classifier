"""Model architecture: a fully connected Deep Neural Network (no CNNs).

The network is built with the Keras Sequential API. Every hidden block follows
the same pattern::

    Dense -> BatchNormalization -> ReLU -> Dropout

and the network ends with a softmax output layer.

The architecture is fully controlled by two tuples, ``hidden_units`` and
``dropout_rates``, so it can be changed without touching the building logic.

Example:
    >>> from src.model import build_model
    >>> model = build_model()                                  # default DNN
    >>> model = build_model(hidden_units=(1024, 512, 256),     # custom DNN
    ...                     dropout_rates=(0.4, 0.3, 0.2))

Run a self-test from the project root with::

    python -m src.model
"""

from __future__ import annotations

import logging
import platform
from typing import Sequence, Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Default architecture (edit these to change the model everywhere)
# --------------------------------------------------------------------------- #
DEFAULT_INPUT_SHAPE: Tuple[int, ...] = (784,)
DEFAULT_NUM_CLASSES: int = 10
DEFAULT_HIDDEN_UNITS: Tuple[int, ...] = (512, 256, 128, 64)
DEFAULT_DROPOUT_RATES: Tuple[float, ...] = (0.3, 0.3, 0.2, 0.2)
DEFAULT_LEARNING_RATE: float = 1e-3

# Use with integer labels (shape (N,)), as returned by load_data(one_hot=False).
# For one-hot labels (shape (N, 10)) pass loss="categorical_crossentropy".
DEFAULT_LOSS: str = "sparse_categorical_crossentropy"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _validate_architecture(
    hidden_units: Sequence[int], dropout_rates: Sequence[float]
) -> None:
    """Check that the architecture arguments are consistent.

    Args:
        hidden_units: Number of neurons in each hidden Dense layer.
        dropout_rates: Dropout rate applied after each hidden layer.

    Raises:
        ValueError: If the sequences are empty, have different lengths, or
            contain invalid values.
    """
    if len(hidden_units) == 0:
        raise ValueError("hidden_units must contain at least one layer.")
    if len(hidden_units) != len(dropout_rates):
        raise ValueError(
            "hidden_units and dropout_rates must have the same length, got "
            f"{len(hidden_units)} and {len(dropout_rates)}."
        )
    if any(units <= 0 for units in hidden_units):
        raise ValueError(f"All layer sizes must be positive, got {hidden_units}.")
    if any(not 0.0 <= rate < 1.0 for rate in dropout_rates):
        raise ValueError(
            f"All dropout rates must be in [0, 1), got {dropout_rates}."
        )


def _get_optimizer(learning_rate: float) -> keras.optimizers.Optimizer:
    """Create the Adam optimizer.

    On Apple Silicon (M1/M2/M3/M4) the new Keras optimizers run slowly under
    ``tensorflow-macos``; the legacy implementation is the recommended
    workaround. Everywhere else the standard Adam is used.

    Args:
        learning_rate: Initial learning rate.

    Returns:
        A configured Adam optimizer.
    """
    is_apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"
    legacy = getattr(keras.optimizers, "legacy", None)
    if is_apple_silicon and legacy is not None:
        logger.info("Apple Silicon detected: using legacy Adam optimizer.")
        return legacy.Adam(learning_rate=learning_rate)
    return keras.optimizers.Adam(learning_rate=learning_rate)


# --------------------------------------------------------------------------- #
# Model construction
# --------------------------------------------------------------------------- #
def build_model(
    input_shape: Tuple[int, ...] = DEFAULT_INPUT_SHAPE,
    num_classes: int = DEFAULT_NUM_CLASSES,
    hidden_units: Sequence[int] = DEFAULT_HIDDEN_UNITS,
    dropout_rates: Sequence[float] = DEFAULT_DROPOUT_RATES,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    loss: str = DEFAULT_LOSS,
    name: str = "fashion_mnist_dnn",
) -> keras.Model:
    """Build and compile a fully connected DNN classifier.

    Each hidden block is ``Dense -> BatchNorm -> ReLU -> Dropout``. The Dense
    layers use He-normal initialization (suited to ReLU) and no bias term,
    because the following BatchNormalization layer already learns a shift.

    Args:
        input_shape: Shape of one input sample. ``(784,)`` for flattened
            Fashion-MNIST images.
        num_classes: Number of output classes.
        hidden_units: Neurons in each hidden layer, in order.
        dropout_rates: Dropout rate after each hidden layer. Must have the same
            length as ``hidden_units``.
        learning_rate: Learning rate for the Adam optimizer.
        loss: Loss function. Use ``"sparse_categorical_crossentropy"`` for
            integer labels or ``"categorical_crossentropy"`` for one-hot labels.
        name: Name of the Keras model.

    Returns:
        A compiled ``keras.Sequential`` model.

    Raises:
        ValueError: If the architecture arguments are inconsistent.
    """
    _validate_architecture(hidden_units, dropout_rates)

    model = keras.Sequential(name=name)
    model.add(keras.Input(shape=input_shape, name="input"))

    # Hidden blocks: Dense -> BatchNorm -> ReLU -> Dropout
    for i, (units, rate) in enumerate(zip(hidden_units, dropout_rates), start=1):
        model.add(
            layers.Dense(
                units,
                kernel_initializer="he_normal",
                use_bias=False,  # BatchNorm provides the shift
                name=f"dense_{i}",
            )
        )
        model.add(layers.BatchNormalization(name=f"bn_{i}"))
        model.add(layers.Activation("relu", name=f"relu_{i}"))
        model.add(layers.Dropout(rate, name=f"dropout_{i}"))

    # Output layer: one probability per class
    model.add(layers.Dense(num_classes, activation="softmax", name="output"))

    model.compile(
        optimizer=_get_optimizer(learning_rate),
        loss=loss,
        metrics=["accuracy"],
    )

    logger.info(
        "Built model '%s' with %s parameters.", name, f"{model.count_params():,}"
    )
    return model


def print_model_summary(model: keras.Model) -> None:
    """Print the Keras summary followed by a parameter breakdown.

    Args:
        model: A built Keras model.
    """
    model.summary()

    trainable = sum(int(tf.size(w)) for w in model.trainable_weights)
    non_trainable = sum(int(tf.size(w)) for w in model.non_trainable_weights)
    print(f"\nTrainable parameters    : {trainable:,}")
    print(f"Non-trainable parameters: {non_trainable:,} (BatchNorm statistics)")
    print(f"Total parameters        : {trainable + non_trainable:,}")


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
def main() -> None:
    """Build the default model, print its summary, and run sanity checks."""
    import numpy as np

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    # --- Default architecture ---------------------------------------------- #
    print("\n=== Default model ===")
    model = build_model()
    print_model_summary(model)

    # Forward pass on random data: output must be valid probabilities.
    batch = np.random.rand(8, DEFAULT_INPUT_SHAPE[0]).astype("float32")
    probs = model.predict(batch, verbose=0)
    assert probs.shape == (8, DEFAULT_NUM_CLASSES), probs.shape
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5), "Softmax must sum to 1"

    # No convolutional layers allowed in this project.
    assert not any("conv" in layer.__class__.__name__.lower() for layer in model.layers)

    # --- Custom architecture ----------------------------------------------- #
    print("\n=== Custom model (changing the architecture is one line) ===")
    custom = build_model(hidden_units=(256, 128), dropout_rates=(0.2, 0.1))
    print(f"Layers: {[layer.name for layer in custom.layers]}")
    print(f"Parameters: {custom.count_params():,}")

    # --- Input validation --------------------------------------------------- #
    try:
        build_model(hidden_units=(128, 64), dropout_rates=(0.3,))
    except ValueError as err:
        print(f"\nValidation works: {err}")
    else:
        raise AssertionError("Mismatched arguments should raise ValueError")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
