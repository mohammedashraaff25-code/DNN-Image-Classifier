"""Unit tests for src.data_loader."""

import numpy as np
import pytest

from src.data_loader import (
    INPUT_DIM,
    NUM_CLASSES,
    flatten_images,
    load_data,
    normalize_images,
    one_hot_encode,
    stratified_train_val_split,
)


class TestFlattenAndNormalize:
    def test_flatten_shape(self):
        images = np.zeros((5, 28, 28), dtype="uint8")
        assert flatten_images(images).shape == (5, INPUT_DIM)

    def test_flatten_keeps_pixel_order(self):
        images = np.arange(2 * 28 * 28).reshape(2, 28, 28)
        assert np.array_equal(flatten_images(images)[0], images[0].ravel())

    def test_flatten_rejects_wrong_dimensions(self):
        with pytest.raises(ValueError, match="Expected images"):
            flatten_images(np.zeros((5, 784)))

    def test_normalize_range_and_dtype(self):
        images = np.array([[0, 128, 255]], dtype="uint8")
        scaled = normalize_images(images)
        assert scaled.dtype == np.float32
        assert scaled.min() == 0.0
        assert scaled.max() == 1.0
        assert scaled[0, 1] == pytest.approx(128 / 255)


class TestStratifiedSplit:
    @pytest.fixture
    def data(self):
        labels = np.repeat(np.arange(10), 100)  # 1,000 balanced samples
        features = np.arange(len(labels)).reshape(-1, 1)  # unique id per row
        return features, labels

    def test_sizes_and_class_balance(self, data):
        x, y = data
        _, y_train, _, y_val = stratified_train_val_split(x, y, val_split=0.15)
        assert len(y_val) == 150
        assert len(y_train) == 850
        assert (np.bincount(y_val) == 15).all()  # same share of every class

    def test_partition_is_disjoint_and_complete(self, data):
        x, y = data
        x_train, _, x_val, _ = stratified_train_val_split(x, y)
        train_ids, val_ids = set(x_train.ravel()), set(x_val.ravel())
        assert train_ids.isdisjoint(val_ids)
        assert len(train_ids | val_ids) == len(y)

    def test_features_stay_aligned_with_labels(self, data):
        x, y = data
        x_train, y_train, _, _ = stratified_train_val_split(x, y)
        assert np.array_equal(y[x_train.ravel()], y_train)

    def test_same_seed_is_reproducible(self, data):
        x, y = data
        first = stratified_train_val_split(x, y, seed=1)
        second = stratified_train_val_split(x, y, seed=1)
        assert all(np.array_equal(a, b) for a, b in zip(first, second, strict=True))

    def test_different_seed_changes_the_split(self, data):
        x, y = data
        first = stratified_train_val_split(x, y, seed=1)
        second = stratified_train_val_split(x, y, seed=2)
        assert not np.array_equal(first[2], second[2])

    @pytest.mark.parametrize("bad_split", [0.0, 1.0, -0.1, 1.5])
    def test_invalid_val_split_raises(self, data, bad_split):
        x, y = data
        with pytest.raises(ValueError, match="val_split"):
            stratified_train_val_split(x, y, val_split=bad_split)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            stratified_train_val_split(np.zeros((10, 1)), np.zeros(9))


class TestOneHot:
    def test_shape_dtype_and_values(self):
        labels = np.array([0, 3, 9, 3])
        encoded = one_hot_encode(labels)
        assert encoded.shape == (4, NUM_CLASSES)
        assert encoded.dtype == np.float32
        assert (encoded.sum(axis=1) == 1).all()
        assert np.array_equal(encoded.argmax(axis=1), labels)

    def test_rejects_already_encoded_labels(self):
        with pytest.raises(ValueError, match="one-hot"):
            one_hot_encode(np.zeros((4, NUM_CLASSES)))


class TestLoadData:
    def test_default_pipeline(self, fake_dataset):
        x_train, y_train, x_val, y_val, x_test, y_test = load_data()
        assert x_train.shape == (850, INPUT_DIM)
        assert x_val.shape == (150, INPUT_DIM)
        assert x_test.shape == (200, INPUT_DIM)
        assert x_train.dtype == np.float32
        assert x_train.min() >= 0.0 and x_train.max() <= 1.0
        assert y_train.shape == (850,)
        assert len(y_val) == 150 and len(y_test) == 200

    def test_one_hot_labels(self, fake_dataset):
        *_, y_test = load_data(one_hot=True)
        assert y_test.shape == (200, NUM_CLASSES)

    def test_raw_images_when_preprocessing_is_disabled(self, fake_dataset):
        x_train, *_ = load_data(flatten=False, normalize=False)
        assert x_train.shape == (850, 28, 28)
        assert x_train.dtype == np.uint8

    def test_custom_validation_split(self, fake_dataset):
        x_train, _, x_val, *_ = load_data(val_split=0.2)
        assert len(x_val) == 200
        assert len(x_train) == 800

    def test_deterministic_for_a_given_seed(self, fake_dataset):
        first = load_data(seed=7)
        second = load_data(seed=7)
        assert all(np.array_equal(a, b) for a, b in zip(first, second, strict=True))
