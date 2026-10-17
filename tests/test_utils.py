"""Unit tests for src.utils."""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from src.utils import save_figure, set_seed, with_plot_style


def test_set_seed_makes_numpy_and_tensorflow_reproducible():
    set_seed(123)
    first = (np.random.rand(3), tf.random.uniform((3,)).numpy())
    set_seed(123)
    second = (np.random.rand(3), tf.random.uniform((3,)).numpy())
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])


def test_save_figure_creates_folders_and_closes_the_figure(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    target = tmp_path / "nested" / "folder" / "plot.png"

    save_figure(fig, target, dpi=50)

    assert target.exists() and target.stat().st_size > 0
    assert not plt.fignum_exists(fig.number)


def test_with_plot_style_does_not_leak_global_settings():
    @with_plot_style("darkgrid")
    def make_plot():
        return plt.rcParams["axes.facecolor"]

    before = plt.rcParams["axes.facecolor"]
    inside = make_plot()
    after = plt.rcParams["axes.facecolor"]

    assert inside != before  # the style was applied inside the function...
    assert after == before  # ...and restored afterwards
