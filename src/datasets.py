import pandas as pd
import tensorflow as tf

class Datasets:
    def __init__(self, train: tuple[pd.DataFrame, pd.DataFrame] | None, val: tuple[pd.DataFrame, pd.DataFrame] | None, test: tuple[pd.DataFrame, pd.DataFrame] | None, dtype: tuple[tf.DType, tf.DType] | None = None):
        self.train = None
        self.val = None
        self.test = None

        if train:
            self.train =  (
                tf.convert_to_tensor(train[0].values, dtype=dtype[0] if dtype else None),
                tf.convert_to_tensor(train[1].values, dtype=dtype[1] if dtype else None)
            )
        if val:
            self.val =  (
                tf.convert_to_tensor(val[0].values, dtype=dtype[0] if dtype else None),
                tf.convert_to_tensor(val[1].values, dtype=dtype[1] if dtype else None)
            )
        if test:
            self.test =  (
                tf.convert_to_tensor(test[0].values, dtype=dtype[0] if dtype else None),
                tf.convert_to_tensor(test[1].values, dtype=dtype[1] if dtype else None)
            )
