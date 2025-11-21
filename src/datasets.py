import pandas as pd
import tensorflow as tf


class Datasets:
    def __init__(
        self,
        train: tuple[pd.DataFrame, pd.DataFrame] | None,
        val: tuple[pd.DataFrame, pd.DataFrame] | None,
        test: tuple[pd.DataFrame, pd.DataFrame] | None,
        dtype: tuple[tf.DType, tf.DType] | None = None,
    ):
        self.train = None
        self.val = None
        self.test = None

        if train:
            self.train = (
                tf.convert_to_tensor(
                    train[0].values, dtype=dtype[0] if dtype else None
                ),
                tf.convert_to_tensor(
                    train[1].values, dtype=dtype[1] if dtype else None
                ),
            )
        if val:
            self.val = (
                tf.convert_to_tensor(val[0].values, dtype=dtype[0] if dtype else None),
                tf.convert_to_tensor(val[1].values, dtype=dtype[1] if dtype else None),
            )
        if test:
            self.test = (
                tf.convert_to_tensor(test[0].values, dtype=dtype[0] if dtype else None),
                tf.convert_to_tensor(test[1].values, dtype=dtype[1] if dtype else None),
            )

    @property
    def X_train(self):
        return self.train[0] if self.train is not None else None

    @property
    def y_train(self):
        return self.train[1] if self.train is not None else None

    @property
    def X_val(self):
        return self.val[0] if self.val is not None else None

    @property
    def y_val(self):
        return self.val[1] if self.val is not None else None

    @property
    def X_test(self):
        return self.test[0] if self.test is not None else None

    @property
    def y_test(self):
        return self.test[1] if self.test is not None else None

    @property
    def train_size(self):
        return int(self.X_train.shape[0]) if self.X_train is not None else 0

    @property
    def val_size(self):
        return int(self.X_val.shape[0]) if self.X_val is not None else 0

    @property
    def test_size(self):
        return int(self.X_test.shape[0]) if self.X_test is not None else 0
