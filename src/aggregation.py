import numpy as np


# Federated average
def fed_avg(flat_models: list[tuple[int, list]]):
    weights = np.zeros_like(flat_models[0][1], dtype=np.float32)
    samples = np.float32(0)

    for n, w in flat_models:
        w = np.asarray(w, dtype=np.float32)

        weights += np.float32(n) * w
        samples += np.float32(n)
    return (weights / samples).astype(np.float32)


# Federated average similarity
def fed_avg_sim(base_model, flat_models, alpha=1.5, min_threshold=0.001):
    local_samples = base_model[0]
    local_vector = np.asarray(base_model[1])
    local_norm = np.linalg.norm(local_vector) + 1e-10

    weighted_vector_sum = local_vector * local_samples
    total_score = local_samples

    for peer in flat_models:
        peer_samples = peer[0]
        peer_vector = np.asarray(peer[1])
        peer_norm = np.linalg.norm(peer_vector) + 1e-10

        # CosSim = (A • B) / (||A|| * ||B||)
        cosine_sim = np.dot(local_vector, peer_vector) / (local_norm * peer_norm)

        if cosine_sim < min_threshold:
            print("Peer rejected", cosine_sim)
            continue

        adjusted_sim = np.power(np.clip(cosine_sim, 0, 1), alpha)

        contribution = peer_samples * adjusted_sim

        weighted_vector_sum += peer_vector * contribution
        total_score += contribution

    target_vector = weighted_vector_sum / total_score
    return target_vector
