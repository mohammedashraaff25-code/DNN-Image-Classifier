"""Small shared helpers: reproducibility and figure saving."""

from __future__ import annotations

import functools
import logging
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and TensorFlow for reproducible runs.

    Args:
        seed: The seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def save_figure(fig: Figure, path: Path, dpi: int = 300) -> None:
    """Save a figure to disk and close it to free memory.

    The parent folder is created if it does not exist.

    Args:
        fig: The matplotlib figure to save.
        path: Destination image path.
        dpi: Resolution of the saved image.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


def with_plot_style(
    style: str = "whitegrid", context: str = "notebook"
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator: run a plotting function inside a temporary seaborn style.

    Unlike ``sns.set_theme``, the style is restored afterwards, so importing and
    calling library functions never changes the caller's global plot settings.

    Args:
        style: Seaborn axes style, e.g. ``"white"`` or ``"whitegrid"``.
        context: Seaborn plotting context, e.g. ``"notebook"``.

    Returns:
        A decorator for plotting functions.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with sns.axes_style(style), sns.plotting_context(context):
                return func(*args, **kwargs)

        return wrapper

    return decorator
