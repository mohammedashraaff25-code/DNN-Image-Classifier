"""Inference: load the trained model and classify clothing images.

Two usage modes:

1. **Your own images** (PNG / JPG of a clothing item)::

       python -m src.predict --image path/to/shirt.jpg
       python -m src.predict --image a.png b.png --top-k 3

2. **Random test-set samples** (default when no image is given)::

       python -m src.predict --test-samples 10

Fashion-MNIST images are *light items on a dark background*. Real photos are
usually the opposite, so custom images are converted automatically:
grayscale -> padded to a square -> resized to 28x28 -> colors inverted if the
background is bright (control with ``--invert auto|yes|no``).

A figure with the predictions is saved to ``reports/figures/12_predictions.png``.

Note:
    The model was trained on small, centered, studio-style product pictures.
    Photos with cluttered backgrounds or unusual angles will be less accurate.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

from src.config import BEST_MODEL_PATH, FIGURES_DIR
from src.data_loader import CLASS_NAMES, IMG_HEIGHT, IMG_WIDTH, load_data
from src.model import load_best_model
from src.utils import save_figure, with_plot_style

logger = logging.getLogger(__name__)

# Output path (shared locations live in src.config)
PREDICTIONS_FIG_PATH: Path = FIGURES_DIR / "12_predictions.png"

# Type alias: list of (class_name, probability)
Prediction = tuple[str, float]


# --------------------------------------------------------------------------- #
# Model and image handling
# --------------------------------------------------------------------------- #
def load_image(image_path: Path, invert: str = "auto") -> np.ndarray:
    """Load an image file and convert it to the Fashion-MNIST format.

    Steps: grayscale -> pad to a square (keeps the aspect ratio) -> resize to
    28x28 -> scale to ``[0, 1]`` -> optionally invert colors.

    Args:
        image_path: Path to a PNG / JPG image.
        invert: ``"auto"`` inverts when the image background is bright,
            ``"yes"`` always inverts, ``"no"`` never inverts.

    Returns:
        ``float32`` array of shape ``(28, 28)`` with values in ``[0, 1]``.

    Raises:
        FileNotFoundError: If the image does not exist.
        ValueError: If ``invert`` is not one of ``auto``, ``yes``, ``no``.
    """
    from PIL import Image  # imported lazily: only needed for custom images

    if invert not in {"auto", "yes", "no"}:
        raise ValueError(f"invert must be 'auto', 'yes' or 'no', got '{invert}'.")
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    gray = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32)

    # Estimate the background brightness from the image border.
    border = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])
    background = float(np.median(border))

    # Pad to a square using the background color so the item is not stretched.
    height, width = gray.shape
    side = max(height, width)
    canvas = np.full((side, side), background, dtype=np.float32)
    top, left = (side - height) // 2, (side - width) // 2
    canvas[top : top + height, left : left + width] = gray

    resized = Image.fromarray(canvas.astype(np.uint8)).resize(
        (IMG_WIDTH, IMG_HEIGHT), Image.Resampling.LANCZOS
    )
    image = np.asarray(resized, dtype=np.float32) / 255.0

    should_invert = background > 127 if invert == "auto" else invert == "yes"
    return 1.0 - image if should_invert else image


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #
def predict_probabilities(model: keras.Model, images: np.ndarray) -> np.ndarray:
    """Predict class probabilities for a batch of images.

    Args:
        model: Trained Keras model.
        images: Array of shape ``(N, 28, 28)`` or ``(N, 784)`` with values in
            ``[0, 1]``.

    Returns:
        Probability array of shape ``(N, 10)``.
    """
    flat = images.reshape(len(images), -1).astype("float32")
    return model.predict(flat, verbose=0)


def top_k_predictions(probabilities: np.ndarray, k: int = 3) -> list[Prediction]:
    """Return the ``k`` most likely classes for one probability vector.

    Args:
        probabilities: Array of shape ``(10,)``.
        k: Number of predictions to return.

    Returns:
        A list of ``(class_name, probability)`` sorted from most to least likely.
    """
    order = np.argsort(probabilities)[::-1][:k]
    return [(CLASS_NAMES[i], float(probabilities[i])) for i in order]


def predict_image(
    model: keras.Model, image_path: Path, top_k: int = 3, invert: str = "auto"
) -> list[Prediction]:
    """Classify a single image file.

    Args:
        model: Trained Keras model.
        image_path: Path to the image.
        top_k: Number of predictions to return.
        invert: Color inversion mode (see :func:`load_image`).

    Returns:
        The top-``k`` ``(class_name, probability)`` predictions.
    """
    image = load_image(image_path, invert=invert)
    probabilities = predict_probabilities(model, image[np.newaxis])[0]
    return top_k_predictions(probabilities, top_k)


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
@with_plot_style("white")
def plot_predictions(
    images: np.ndarray,
    probabilities: np.ndarray,
    save_path: Path,
    true_labels: np.ndarray | None = None,
    names: Sequence[str] | None = None,
) -> None:
    """Plot a grid of images with their predicted class and confidence.

    When ``true_labels`` are given, correct predictions are green and wrong
    ones are red.

    Args:
        images: Array of shape ``(N, 28, 28)`` or ``(N, 784)``.
        probabilities: Probability array of shape ``(N, 10)``.
        save_path: Destination image path.
        true_labels: Optional ground-truth labels of shape ``(N,)``.
        names: Optional display name for each image (e.g. the file name).
    """
    n_images = len(images)
    n_cols = min(5, n_images)
    n_rows = int(np.ceil(n_images / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.8 * n_cols, 3.3 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for i, ax in enumerate(axes[:n_images]):
        predicted = int(probabilities[i].argmax())
        confidence = float(probabilities[i, predicted])

        ax.imshow(images[i].reshape(IMG_HEIGHT, IMG_WIDTH), cmap="gray")
        ax.axis("off")

        lines = [f"Pred: {CLASS_NAMES[predicted]} ({confidence:.0%})"]
        color = "black"
        if true_labels is not None:
            lines.append(f"True: {CLASS_NAMES[int(true_labels[i])]}")
            color = "seagreen" if predicted == int(true_labels[i]) else "firebrick"
        elif names is not None:
            lines.append(str(names[i]))
        ax.set_title("\n".join(lines), fontsize=10, color=color)

    for ax in axes[n_images:]:
        ax.axis("off")

    fig.suptitle("Model predictions", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, save_path)


# --------------------------------------------------------------------------- #
# Command line interface
# --------------------------------------------------------------------------- #
def _print_predictions(label: str, predictions: list[Prediction]) -> None:
    """Print the top-k predictions of one image in a compact format."""
    print(f"\n{label}")
    for rank, (name, probability) in enumerate(predictions, start=1):
        print(f"  {rank}. {name:<12} {probability:6.2%}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Classify clothing images with the trained Fashion-MNIST DNN."
    )
    parser.add_argument(
        "--image",
        type=Path,
        nargs="+",
        help="One or more image files to classify.",
    )
    parser.add_argument(
        "--test-samples",
        type=int,
        default=10,
        help="Number of random test images to classify when --image is not given.",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--invert",
        choices=["auto", "yes", "no"],
        default="auto",
        help="Invert colors of custom images (light item on dark background).",
    )
    parser.add_argument("--model-path", type=Path, default=BEST_MODEL_PATH)
    parser.add_argument("--save-path", type=Path, default=PREDICTIONS_FIG_PATH)
    parser.add_argument("--seed", type=int, default=None, help="Seed for test sampling.")
    return parser.parse_args()


def main() -> None:
    """Entry point: classify custom images or random test samples."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    plt.switch_backend("Agg")  # headless: figures are saved, never shown
    args = parse_args()
    model = load_best_model(args.model_path)

    if args.image:
        # --- Custom images ------------------------------------------------- #
        images = np.stack([load_image(path, args.invert) for path in args.image])
        probabilities = predict_probabilities(model, images)
        names = [path.name for path in args.image]

        for name, probs in zip(names, probabilities, strict=True):
            _print_predictions(name, top_k_predictions(probs, args.top_k))
        plot_predictions(images, probabilities, args.save_path, names=names)

    else:
        # --- Random test samples ------------------------------------------- #
        _, _, _, _, X_test, y_test = load_data(one_hot=False)
        rng = np.random.default_rng(args.seed)
        n_samples = min(args.test_samples, len(X_test))
        indices = rng.choice(len(X_test), size=n_samples, replace=False)

        images, true_labels = X_test[indices], y_test[indices]
        probabilities = predict_probabilities(model, images)
        predicted = probabilities.argmax(axis=1)

        for i, idx in enumerate(indices):
            verdict = "OK " if predicted[i] == true_labels[i] else "ERR"
            _print_predictions(
                f"[{verdict}] test image #{idx} (true: {CLASS_NAMES[true_labels[i]]})",
                top_k_predictions(probabilities[i], args.top_k),
            )
        accuracy = float((predicted == true_labels).mean())
        print(f"\nAccuracy on these {n_samples} samples: {accuracy:.0%}")
        plot_predictions(images, probabilities, args.save_path, true_labels=true_labels)

    print(f"\nFigure saved to {args.save_path}")


if __name__ == "__main__":
    main()
