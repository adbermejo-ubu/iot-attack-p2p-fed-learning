import numpy as np
from keras import layers, regularizers, Model, optimizers, metrics, initializers

SEED = 42


# Load the model
def load_model(dim: int, name: str = "mlp"):
    init = initializers.GlorotUniform(seed=SEED)
    bias_init = initializers.Zeros()

    # Input layer
    inputs = layers.Input(shape=(dim,), name="input_layer")
    # Hidden layer 1
    x = layers.Dense(
        512,
        activation="relu",
        kernel_initializer=init,
        bias_initializer=bias_init,
        kernel_regularizer=regularizers.l2(0.001),
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    # Hidden layer 2
    x = layers.Dense(
        256,
        activation="relu",
        kernel_initializer=init,
        bias_initializer=bias_init,
        kernel_regularizer=regularizers.l2(0.001),
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.35)(x)
    # Hidden layer 3
    x = layers.Dense(
        128,
        activation="relu",
        kernel_initializer=init,
        bias_initializer=bias_init,
        kernel_regularizer=regularizers.l2(0.001),
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    # Hidden layer 4
    x = layers.Dense(
        64, activation="relu", kernel_initializer=init, bias_initializer=bias_init
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)
    # Output layer
    outputs = layers.Dense(
        1,
        activation="sigmoid",
        kernel_initializer=init,
        bias_initializer=bias_init,
        name="output_layer",
    )(x)
    model = Model(inputs=inputs, outputs=outputs, name=name)

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


# Flat the model to a single one dimension array
def flat_model(model: Model):
    return np.concatenate(
        [np.array(w.numpy(), dtype=np.float32).ravel() for w in model.weights],
        dtype=np.float32,
    ).astype(np.float32)


# Reconstruct model
def reshape_model(
    flat_model: list,
    shapes: list,
    sizes: list,
):
    flat_model = np.asarray(flat_model, dtype=np.float32)

    if len(flat_model) != sum(sizes):
        return

    weights = []
    pointer = 0

    for shape, size in zip(shapes, sizes):
        chunk = flat_model[pointer : pointer + size]
        reshaped_chunk = chunk.reshape(shape)

        weights.append(reshaped_chunk)
        pointer += size
    return weights


# Federated average
def fed_avg(flat_models: list[tuple[int, list]]):
    weights = np.zeros_like(flat_models[0][1], dtype=np.float32)
    samples = 0

    for n, w in flat_models:
        w = np.asarray(w, dtype=np.float32)

        weights += np.float32(n) * w
        samples += int(n)
    return (weights / np.float32(samples)).astype(np.float32)

# Federated average similarity
def fed_avg_sim(base_model: tuple[int, list], flat_models: list[tuple[int, list]], alpha: float = 0.5):
    total = np.float32(base_model[0])
    weights = total * np.copy(base_model[1])
    
    for n, w in flat_models:
        sim = np.exp(-alpha * np.linalg.norm(w - base_model[1], ord=2)) # Similarity with the base model
        cont = sim * np.float32(n)

        weights += cont * np.asarray(w, dtype=np.float32)
        total += cont
    return (weights / total).astype(np.float32)
