"""Training pipeline for the Fashion-MNIST fully connected DNN.

The script performs the following steps:

1. Load and preprocess Fashion-MNIST (``src.data_loader``).
2. Build and compile the model (``src.model``).
3. Train with three callbacks that fight overfitting:
   ``EarlyStopping``, ``ReduceLROnPlateau`` and ``ModelCheckpoint``.
4. Save the training history to ``reports/training_history.csv``.
5. Plot loss / accuracy curves to ``reports/figures/07_training_curves.png``.
6. Save the best model to ``models/best_model.keras`` and report its
   validation and test performance.

Run from the project root::

    python -m src.train
    python -m src.train --epochs 150 --batch-size 256 --learning-rate 0.0005

All output paths are resolved relative to the project root, so the script works
no matter which directory it is launched from.
"""

from __future__ import annotations

import argparse
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")  # headless backend: figures are saved, never shown

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
import tensorflow as tf  # noqa: E402
from tensorflow import keras  # noqa: E402

from src.data_loader import load_data  # noqa: E402
from src.model import build_model, print_model_summary  # noqa: E402

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Paths (relative to the project root, not to the current working directory)
# --------------------------------------------------------------------------- #
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"

BEST_MODEL_PATH: Path = MODELS_DIR / "best_model.keras"
HISTORY_PATH: Path = REPORTS_DIR / "training_history.csv"
CURVES_PATH: Path = FIGURES_DIR / "07_training_curves.png"


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    """Hyperparameters and settings for a training run.

    Attributes:
        epochs: Maximum number of epochs (EarlyStopping usually stops earlier).
        batch_size: Mini-batch size.
        learning_rate: Initial Adam learning rate.
        val_split: Fraction of the training set held out for validation.
        seed: Random seed for reproducibility.
        es_patience: Epochs without val_loss improvement before stopping.
        lr_patience: Epochs without improvement before reducing the LR.
        lr_factor: Multiplier applied to the LR when it is reduced.
        min_lr: Lower bound for the learning rate.
    """

    epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 1e-3
    val_split: float = 0.15
    seed: int = 42
    es_patience: int = 12
    lr_patience: int = 5
    lr_factor: float = 0.5
    min_lr: float = 1e-6


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    """Seed Python, NumPy and TensorFlow for reproducible runs.

    Args:
        seed: The seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def create_callbacks(
    config: TrainConfig, checkpoint_path: Path = BEST_MODEL_PATH
) -> List[keras.callbacks.Callback]:
    """Create the callbacks used to prevent overfitting.

    * **EarlyStopping** stops training when ``val_loss`` stops improving and
      restores the weights of the best epoch.
    * **ReduceLROnPlateau** halves the learning rate when ``val_loss`` plateaus,
      which lets the optimizer settle into a better minimum. Its patience is
      shorter than EarlyStopping's so the LR is reduced *before* we give up.
    * **ModelCheckpoint** writes the best model (lowest ``val_loss``) to disk
      every time it improves, so the best model is safe even if the run is
      interrupted.

    Args:
        config: Training configuration.
        checkpoint_path: Where ModelCheckpoint saves the best model.

    Returns:
        A list of Keras callbacks.
    """
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.es_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.lr_factor,
            patience=config.lr_patience,
            min_lr=config.min_lr,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
    ]


def save_history(history: Dict[str, List[float]], path: Path = HISTORY_PATH) -> None:
    """Save the per-epoch training history to a CSV file.

    Args:
        history: ``History.history`` dictionary returned by ``model.fit``.
        path: Destination CSV path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(history)
    df.insert(0, "epoch", np.arange(1, len(df) + 1))
    df.to_csv(path, index=False)
    logger.info("Saved training history to %s", path)


def plot_training_curves(
    history: Dict[str, List[float]], save_path: Path = CURVES_PATH
) -> None:
    """Plot and save loss, accuracy (and learning rate) curves.

    Train and validation curves are drawn together so overfitting shows up as
    a growing gap between them. The best epoch is marked with a dashed line.

    Args:
        history: ``History.history`` dictionary returned by ``model.fit``.
        save_path: Destination image path.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = np.arange(1, len(history["loss"]) + 1)
    best_epoch = int(np.argmin(history["val_loss"])) + 1

    # The LR key is "lr" in Keras 2 and "learning_rate" in Keras 3.
    lr_key = next((k for k in ("lr", "learning_rate") if k in history), None)
    n_panels = 3 if lr_key else 2

    sns.set_theme(style="whitegrid", context="notebook")
    train_color, val_color = sns.color_palette("deep")[0], sns.color_palette("deep")[3]
    fig, axes = plt.subplots(1, n_panels, figsize=(6.2 * n_panels, 4.8))

    panels = [
        ("loss", "Loss", "Loss curves"),
        ("accuracy", "Accuracy", "Accuracy curves"),
    ]
    for ax, (metric, ylabel, title) in zip(axes, panels):
        ax.plot(epochs, history[metric], color=train_color, linewidth=2, label="Train")
        ax.plot(
            epochs, history[f"val_{metric}"], color=val_color, linewidth=2, label="Validation"
        )
        ax.axvline(
            best_epoch, color="gray", linestyle="--", linewidth=1.2,
            label=f"Best epoch ({best_epoch})",
        )
        ax.set_title(f"{title}: Train vs Validation", fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend()

    if lr_key:
        ax = axes[2]
        ax.plot(epochs, history[lr_key], color=sns.color_palette("deep")[2], linewidth=2)
        ax.set_yscale("log")
        ax.set_title("Learning rate schedule", fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning rate (log)")

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved training curves to %s", save_path)


def evaluate_best_model(
    model_path: Path,
    splits: Dict[str, Tuple[np.ndarray, np.ndarray]],
    batch_size: int,
) -> Dict[str, Dict[str, float]]:
    """Reload the best saved model and evaluate it on every split.

    Reloading from disk (instead of using the in-memory model) guarantees we
    report the performance of exactly the file that was saved.

    Args:
        model_path: Path to the saved ``.keras`` model.
        splits: Mapping ``{name: (X, y)}`` of the splits to evaluate.
        batch_size: Batch size used for evaluation.

    Returns:
        Mapping ``{name: {"loss": ..., "accuracy": ...}}``.
    """
    best_model = keras.models.load_model(model_path)
    results: Dict[str, Dict[str, float]] = {}
    for name, (x, y) in splits.items():
        loss, accuracy = best_model.evaluate(x, y, batch_size=batch_size, verbose=0)
        results[name] = {"loss": float(loss), "accuracy": float(accuracy)}
    return results


def print_final_report(
    history: Dict[str, List[float]],
    results: Dict[str, Dict[str, float]],
    elapsed_seconds: float,
) -> None:
    """Print a readable end-of-training report.

    Args:
        history: ``History.history`` dictionary returned by ``model.fit``.
        results: Output of :func:`evaluate_best_model`.
        elapsed_seconds: Total training time.
    """
    best_idx = int(np.argmin(history["val_loss"]))
    train_acc = history["accuracy"][best_idx]
    val_acc = history["val_accuracy"][best_idx]
    gap = train_acc - val_acc

    line = "=" * 60
    print(f"\n{line}\nTRAINING COMPLETE\n{line}")
    print(f"Epochs run            : {len(history['loss'])}")
    print(f"Best epoch            : {best_idx + 1}")
    print(f"Training time         : {elapsed_seconds / 60:.1f} min")
    print(f"\nAt best epoch:")
    print(f"  Train accuracy      : {train_acc:.4f}")
    print(f"  Validation accuracy : {val_acc:.4f}")
    print(f"  Overfitting gap     : {gap:+.4f} (train - val)")
    print(f"\nBest model (reloaded from {BEST_MODEL_PATH.name}):")
    for name, metrics in results.items():
        print(
            f"  {name.capitalize():<11} loss={metrics['loss']:.4f}  "
            f"accuracy={metrics['accuracy']:.4f}"
        )
    print(f"\nSaved model   : {BEST_MODEL_PATH}")
    print(f"Saved history : {HISTORY_PATH}")
    print(f"Saved curves  : {CURVES_PATH}")
    print(line)


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #
def run_training(config: TrainConfig) -> Dict[str, List[float]]:
    """Run the full training pipeline.

    Args:
        config: Training configuration.

    Returns:
        The Keras training history dictionary.
    """
    set_seed(config.seed)
    for directory in (MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    # 1) Data -------------------------------------------------------------- #
    print("\n[1/5] Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(
        val_split=config.val_split, one_hot=False, seed=config.seed
    )
    print(f"      train={X_train.shape}  val={X_val.shape}  test={X_test.shape}")

    # 2) Model ------------------------------------------------------------- #
    print("\n[2/5] Building model...")
    model = build_model(
        input_shape=(X_train.shape[1],),
        learning_rate=config.learning_rate,
        loss="sparse_categorical_crossentropy",  # integer labels
    )
    print_model_summary(model)

    # 3) Training ---------------------------------------------------------- #
    print("\n[3/5] Training...")
    start = time.time()
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=create_callbacks(config),
        shuffle=True,
        verbose=2,  # one line per epoch
    )
    elapsed = time.time() - start

    # 4) Save history + plots --------------------------------------------- #
    print("\n[4/5] Saving history and plots...")
    save_history(history.history)
    plot_training_curves(history.history)

    # 5) Evaluate the best saved model ------------------------------------ #
    print("\n[5/5] Evaluating best model...")
    results = evaluate_best_model(
        BEST_MODEL_PATH,
        {
            "train": (X_train, y_train),
            "validation": (X_val, y_val),
            "test": (X_test, y_test),
        },
        batch_size=config.batch_size,
    )
    print_final_report(history.history, results, elapsed)

    return history.history


def parse_args() -> TrainConfig:
    """Parse command line arguments into a :class:`TrainConfig`.

    Returns:
        The training configuration.
    """
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(
        description="Train the Fashion-MNIST fully connected DNN."
    )
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--val-split", type=float, default=defaults.val_split)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument(
        "--patience", type=int, default=defaults.es_patience,
        help="EarlyStopping patience (epochs).",
    )
    args = parser.parse_args()

    return TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        val_split=args.val_split,
        seed=args.seed,
        es_patience=args.patience,
    )


def main() -> None:
    """Entry point: parse arguments and run training."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    run_training(parse_args())


if __name__ == "__main__":
    main()
