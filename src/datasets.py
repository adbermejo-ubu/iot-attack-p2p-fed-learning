import os
import pandas as pd
import tensorflow as tf

# Load all datasets
def load_datasets(
    train_path: tuple[str, str] | None,
    val_path: tuple[str, str] | None,
    test_path: tuple[str, str] | None,
) -> tuple[
    tuple[tf.Tensor, tf.Tensor] | None,
    tuple[tf.Tensor, tf.Tensor] | None,
    tuple[tf.Tensor, tf.Tensor] | None,
]:
    def load_split(path: tuple[str, str] | None):
        try:
            return (
                tf.convert_to_tensor(
                    pd.read_csv(os.path.abspath(path[0])).values, dtype=tf.float32
                ),
                tf.convert_to_tensor(
                    pd.read_csv(os.path.abspath(path[1])).values, dtype=tf.int8
                ),
            )
        except:
            return None

    return (load_split(train_path), load_split(val_path), load_split(test_path))
