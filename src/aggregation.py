import numpy as np

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
