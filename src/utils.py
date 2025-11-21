import os
import pandas as pd
from src.model import P2PModel
from src.datasets import Datasets
from keras import layers, regularizers, optimizers, metrics


# Load the model
def load_model(dim: int, name: str = "mlp"):
    # Input layer
    inputs = layers.Input(shape=(dim,), name="input_layer")
    # Hidden layer 1
    x = layers.Dense(512, activation="relu", kernel_regularizer=regularizers.l2(0.001))(
        inputs
    )
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    # Hidden layer 2
    x = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(0.001))(
        x
    )
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.35)(x)
    # Hidden layer 3
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.001))(
        x
    )
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    # Hidden layer 4
    x = layers.Dense(64, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)
    # Output layer
    outputs = layers.Dense(1, activation="sigmoid", name="output_layer")(x)
    model = P2PModel(inputs=inputs, outputs=outputs, name=name)

    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.0003),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            metrics.Precision(name="precision"),
            metrics.Recall(name="recall"),
            metrics.AUC(name="auc"),
        ],
    )
    return model


# Load the datasets
def load_datasets(train: tuple[str, str], val: tuple[str, str], test: tuple[str, str]):
    return Datasets(
        (
            pd.read_csv(os.path.abspath(train[0])),
            pd.read_csv(os.path.abspath(train[1])),
        ),
        (pd.read_csv(os.path.abspath(val[0])), pd.read_csv(os.path.abspath(val[1]))),
        (pd.read_csv(os.path.abspath(test[0])), pd.read_csv(os.path.abspath(test[1]))),
    )
