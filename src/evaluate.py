"""Evaluation of the trained Fashion-MNIST fully connected DNN.

The script loads the best saved model (``models/best_model.keras``), evaluates it
on the held-out **test set**, and produces:

* Test loss and test accuracy (printed and saved)
* A full per-class classification report (precision, recall, F1-score)
* A confusion matrix (raw counts and row-normalized)
* A per-class precision / recall / F1 bar chart
* A grid of misclassified examples
* The most frequently confused class pairs

Outputs are written to the ``reports/`` folder:

====================================  =========================================
``reports/evaluation_metrics.json``   Loss, accuracy, F1 scores, top confusions
``reports/classification_report.txt`` Human-readable classification report
``reports/classification_report.csv`` Per-class metrics as a table
``reports/figures/08_*.png``          Confusion matrix (counts)
``reports/figures/09_*.png``          Confusion matrix (normalized)
``reports/figures/10_*.png``          Per-class precision / recall / F1
``reports/figures/11_*.png``          Misclassified examples
====================================  =========================================

Run from the project root (after training)::

    python -m src.evaluate
    python -m src.evaluate --n-misclassified 20 --misclassified-mode random
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow import keras

from src.config import BEST_MODEL_PATH, FIGURES_DIR, REPORTS_DIR
from src.data_loader import CLASS_NAMES, IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES, load_data
from src.model import load_best_model
from src.utils import save_figure, with_plot_style

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Output file names (folders live in src.config)
# --------------------------------------------------------------------------- #
METRICS_JSON_NAME = "evaluation_metrics.json"
REPORT_TXT_NAME = "classification_report.txt"
REPORT_CSV_NAME = "classification_report.csv"
CM_COUNTS_NAME = "08_confusion_matrix_counts.png"
CM_NORMALIZED_NAME = "09_confusion_matrix_normalized.png"
PER_CLASS_NAME = "10_per_class_metrics.png"
MISCLASSIFIED_NAME = "11_misclassified_examples.png"


# --------------------------------------------------------------------------- #
# Loading and prediction
# --------------------------------------------------------------------------- #
def predict(
    model: keras.Model, x: np.ndarray, batch_size: int = 256
) -> tuple[np.ndarray, np.ndarray]:
    """Predict class probabilities and labels.

    Args:
        model: Trained Keras model.
        x: Input array of shape ``(N, 784)``.
        batch_size: Prediction batch size.

    Returns:
        A tuple ``(y_pred, probabilities)`` with shapes ``(N,)`` and ``(N, 10)``.
    """
    probabilities = model.predict(x, batch_size=batch_size, verbose=0)
    return probabilities.argmax(axis=1), probabilities


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compute_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[str, dict]:
    """Compute the per-class classification report.

    Args:
        y_true: Ground-truth integer labels.
        y_pred: Predicted integer labels.

    Returns:
        A tuple ``(text_report, report_dict)``.
    """
    kwargs = {
        "y_true": y_true,
        "y_pred": y_pred,
        "labels": list(range(NUM_CLASSES)),
        "target_names": CLASS_NAMES,
        "digits": 4,
        "zero_division": 0,
    }
    text = classification_report(**kwargs)
    as_dict = classification_report(**kwargs, output_dict=True)
    return text, as_dict


def find_top_confusions(cm: np.ndarray, top_k: int = 5) -> list[dict[str, object]]:
    """Find the most frequent misclassification pairs.

    Args:
        cm: Confusion matrix of shape ``(C, C)`` (rows = true, columns = predicted).
        top_k: Number of pairs to return.

    Returns:
        A list of dicts ``{"true", "predicted", "count"}`` sorted by count.
    """
    off_diagonal = cm.copy()
    np.fill_diagonal(off_diagonal, 0)

    flat_order = np.argsort(off_diagonal, axis=None)[::-1][:top_k]
    pairs = []
    for flat_index in flat_order:
        true_id, pred_id = np.unravel_index(flat_index, off_diagonal.shape)
        pairs.append(
            {
                "true": CLASS_NAMES[true_id],
                "predicted": CLASS_NAMES[pred_id],
                "count": int(off_diagonal[true_id, pred_id]),
            }
        )
    return pairs


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
@with_plot_style("white")
def plot_confusion_matrix(
    cm: np.ndarray, save_path: Path, normalize: bool = False, accuracy: float | None = None
) -> None:
    """Plot and save a confusion matrix heatmap.

    Args:
        cm: Confusion matrix of shape ``(C, C)``.
        save_path: Destination image path.
        normalize: If ``True``, each row is divided by its sum so cells show the
            share of each true class (the diagonal is then the per-class recall).
        accuracy: Optional overall accuracy shown in the title.
    """
    if normalize:
        data = cm / cm.sum(axis=1, keepdims=True)
        fmt, cbar_label, kind = ".1%", "Share of true class", "normalized"
    else:
        data, fmt, cbar_label, kind = cm, "d", "Number of samples", "counts"

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        data,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        square=True,
        linewidths=0.5,
        linecolor="white",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cbar_kws={"label": cbar_label, "shrink": 0.8},
        annot_kws={"fontsize": 9},
        ax=ax,
    )

    title = f"Confusion matrix ({kind})"
    if accuracy is not None:
        title += f"  |  Test accuracy: {accuracy:.2%}"
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)

    fig.tight_layout()
    save_figure(fig, save_path)


@with_plot_style("whitegrid")
def plot_per_class_metrics(report: dict, save_path: Path) -> None:
    """Plot precision, recall and F1-score for every class.

    Args:
        report: Dictionary returned by ``classification_report(output_dict=True)``.
        save_path: Destination image path.
    """
    rows = [
        {"class": name, "metric": metric, "score": report[name][metric]}
        for name in CLASS_NAMES
        for metric in ("precision", "recall", "f1-score")
    ]
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(13, 5.5))
    sns.barplot(data=df, x="class", y="score", hue="metric", palette="deep", ax=ax)
    ax.set_ylim(0.5, 1.02)
    ax.set_title("Per-class precision, recall and F1-score", fontsize=14, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.legend(title="", loc="lower left", ncol=3)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    fig.tight_layout()
    save_figure(fig, save_path)


@with_plot_style("white")
def plot_misclassified(
    x_test: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    save_path: Path,
    n_examples: int = 20,
    mode: str = "confident",
    seed: int = 42,
) -> None:
    """Plot a grid of misclassified test images.

    Args:
        x_test: Flattened test images of shape ``(N, 784)``.
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        probabilities: Predicted class probabilities of shape ``(N, 10)``.
        save_path: Destination image path.
        n_examples: Number of errors to display (arranged in 5 columns).
        mode: ``"confident"`` shows the errors the model was *most sure* about
            (usually the most informative); ``"random"`` shows a random sample.
        seed: Random seed for ``mode="random"``.
    """
    wrong_idx = np.flatnonzero(y_true != y_pred)
    if len(wrong_idx) == 0:
        logger.info("No misclassified examples to plot.")
        return

    n_examples = min(n_examples, len(wrong_idx))
    if mode == "confident":
        confidence = probabilities[wrong_idx, y_pred[wrong_idx]]
        chosen = wrong_idx[np.argsort(confidence)[::-1][:n_examples]]
        subtitle = "most confident errors"
    elif mode == "random":
        chosen = np.random.default_rng(seed).choice(wrong_idx, n_examples, replace=False)
        subtitle = "random sample of errors"
    else:
        raise ValueError(f"mode must be 'confident' or 'random', got '{mode}'.")

    n_cols = 5
    n_rows = int(np.ceil(n_examples / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.6 * n_cols, 3.0 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, idx in zip(axes, chosen, strict=False):  # spare axes are hidden below
        ax.imshow(x_test[idx].reshape(IMG_HEIGHT, IMG_WIDTH), cmap="gray")
        confidence = probabilities[idx, y_pred[idx]]
        ax.set_title(
            f"True: {CLASS_NAMES[y_true[idx]]}\n"
            f"Pred: {CLASS_NAMES[y_pred[idx]]} ({confidence:.0%})",
            fontsize=9,
            color="firebrick",
        )
        ax.axis("off")
    for ax in axes[len(chosen) :]:
        ax.axis("off")

    fig.suptitle(
        f"Misclassified test examples ({subtitle}; " f"{len(wrong_idx):,} errors in total)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    save_figure(fig, save_path)


# --------------------------------------------------------------------------- #
# Saving results
# --------------------------------------------------------------------------- #
def save_results(
    test_loss: float,
    test_accuracy: float,
    report_text: str,
    report_dict: dict,
    top_confusions: list[dict[str, object]],
    n_test: int,
    n_errors: int,
    reports_dir: Path = REPORTS_DIR,
) -> None:
    """Save metrics as JSON, the text report, and the per-class CSV table.

    Args:
        test_loss: Loss on the test set.
        test_accuracy: Accuracy on the test set.
        report_text: Text classification report.
        report_dict: Classification report as a dictionary.
        top_confusions: Most frequent misclassification pairs.
        n_test: Number of test samples.
        n_errors: Number of misclassified test samples.
        reports_dir: Folder where the report files are written.
    """
    reports_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "test_loss": round(test_loss, 6),
        "test_accuracy": round(test_accuracy, 6),
        "n_test_samples": n_test,
        "n_misclassified": n_errors,
        "macro_f1": round(report_dict["macro avg"]["f1-score"], 6),
        "weighted_f1": round(report_dict["weighted avg"]["f1-score"], 6),
        "per_class": {
            name: {
                "precision": round(report_dict[name]["precision"], 6),
                "recall": round(report_dict[name]["recall"], 6),
                "f1_score": round(report_dict[name]["f1-score"], 6),
                "support": int(report_dict[name]["support"]),
            }
            for name in CLASS_NAMES
        },
        "top_confusions": top_confusions,
    }
    (reports_dir / METRICS_JSON_NAME).write_text(json.dumps(metrics, indent=2))

    header = f"Test loss     : {test_loss:.4f}\n" f"Test accuracy : {test_accuracy:.4f}\n\n"
    (reports_dir / REPORT_TXT_NAME).write_text(header + report_text)

    pd.DataFrame(report_dict).T.round(4).to_csv(reports_dir / REPORT_CSV_NAME, index_label="class")
    logger.info("Saved metrics, text report and CSV table to %s", reports_dir)


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #
def run_evaluation(
    model_path: Path = BEST_MODEL_PATH,
    batch_size: int = 256,
    n_misclassified: int = 20,
    misclassified_mode: str = "confident",
    reports_dir: Path = REPORTS_DIR,
) -> dict[str, float]:
    """Run the complete evaluation pipeline.

    Args:
        model_path: Path to the saved model.
        batch_size: Batch size for evaluation and prediction.
        n_misclassified: Number of misclassified examples to plot.
        misclassified_mode: ``"confident"`` or ``"random"``.
        reports_dir: Folder for the reports; figures go to its ``figures/`` subfolder.

    Returns:
        A dictionary with ``test_loss`` and ``test_accuracy``.
    """
    figures_dir = reports_dir / FIGURES_DIR.relative_to(REPORTS_DIR)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1) Model and data ---------------------------------------------------- #
    print("\n[1/5] Loading model and test data...")
    model = load_best_model(model_path)
    _, _, _, _, X_test, y_test = load_data(one_hot=False)
    print(f"      test set: {X_test.shape}")

    # 2) Loss and accuracy ------------------------------------------------- #
    print("\n[2/5] Evaluating on the test set...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)
    y_pred, probabilities = predict(model, X_test, batch_size)
    recomputed_accuracy = accuracy_score(y_test, y_pred)
    if not np.isclose(recomputed_accuracy, test_accuracy, atol=1e-4):
        logger.warning(
            "Accuracy from model.evaluate (%.4f) differs from the accuracy recomputed "
            "from predictions (%.4f).",
            test_accuracy,
            recomputed_accuracy,
        )

    line = "=" * 60
    print(f"\n{line}\nTEST RESULTS\n{line}")
    print(f"Test loss     : {test_loss:.4f}")
    print(f"Test accuracy : {test_accuracy:.4f} ({test_accuracy:.2%})")

    # 3) Classification report -------------------------------------------- #
    print("\n[3/5] Classification report")
    report_text, report_dict = compute_classification_report(y_test, y_pred)
    print(f"\n{report_text}")

    cm = confusion_matrix(y_test, y_pred, labels=list(range(NUM_CLASSES)))
    top_confusions = find_top_confusions(cm, top_k=5)
    n_errors = int((y_test != y_pred).sum())

    print("Most confused class pairs (true -> predicted):")
    for pair in top_confusions:
        print(f"  {pair['true']:<12} -> {pair['predicted']:<12} {pair['count']:>4} times")

    # 4) Figures ----------------------------------------------------------- #
    print("\n[4/5] Creating figures...")
    plot_confusion_matrix(cm, figures_dir / CM_COUNTS_NAME, normalize=False, accuracy=test_accuracy)
    plot_confusion_matrix(
        cm, figures_dir / CM_NORMALIZED_NAME, normalize=True, accuracy=test_accuracy
    )
    plot_per_class_metrics(report_dict, figures_dir / PER_CLASS_NAME)
    plot_misclassified(
        X_test,
        y_test,
        y_pred,
        probabilities,
        figures_dir / MISCLASSIFIED_NAME,
        n_examples=n_misclassified,
        mode=misclassified_mode,
    )

    # 5) Save results ------------------------------------------------------ #
    print("\n[5/5] Saving results...")
    save_results(
        float(test_loss),
        float(test_accuracy),
        report_text,
        report_dict,
        top_confusions,
        n_test=len(y_test),
        n_errors=n_errors,
        reports_dir=reports_dir,
    )

    print(f"\n{line}")
    print(f"Misclassified : {n_errors:,} of {len(y_test):,} test images")
    print(f"Reports saved : {reports_dir}")
    print(f"Figures saved : {figures_dir}")
    print(line)

    return {"test_loss": float(test_loss), "test_accuracy": float(test_accuracy)}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate the trained Fashion-MNIST DNN on the test set."
    )
    parser.add_argument("--model-path", type=Path, default=BEST_MODEL_PATH)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--n-misclassified",
        type=int,
        default=20,
        help="How many misclassified examples to plot.",
    )
    parser.add_argument(
        "--misclassified-mode",
        choices=["confident", "random"],
        default="confident",
        help="Show the most confident errors or a random sample.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: parse arguments and run the evaluation."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    plt.switch_backend("Agg")  # headless: figures are saved, never shown
    args = parse_args()
    run_evaluation(
        model_path=args.model_path,
        batch_size=args.batch_size,
        n_misclassified=args.n_misclassified,
        misclassified_mode=args.misclassified_mode,
    )


if __name__ == "__main__":
    main()
