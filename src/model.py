import numpy as np
import keras

class P2PModel(keras.Model):
    """Model wrapper for P2P federated learning."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__shapes = [w.shape for w in self.get_weights()]
        self.__sizes = [w.size for w in self.get_weights()]
        self.__params = sum(self.__sizes)

    def flat(self):
        """Reshape the model weights into a one-dimensional array."""
        return np.concatenate([np.array(w.numpy(), dtype=np.float32).ravel() for w in self.weights], dtype=np.float32).astype(np.float32)

    def reconstruct(self, flat_model: list):
        """Reshape one-dimensional array into the model architecture."""
        flat_model = np.asarray(flat_model, dtype=np.float32)
        weights = []
        pointer = 0

        if len(flat_model) !=  self.__params:
            raise ValueError("The flat model does not have the same shape as the model")

        for shape, size in zip(self.__shapes, self.__sizes):
            chunk = flat_model[pointer : pointer + size]
            reshaped_chunk = chunk.reshape(shape)

            weights.append(reshaped_chunk)
            pointer += size
        self.set_weights(weights)

    def segment(self, num: int = 1, exclusions: list[int] = []):
        """Segment the model into the requested number of segments."""
        segments = []

        if not exclusions:
            keys_per_segment = int(np.ceil(self.__params / num))

            current = 0
            while current + keys_per_segment <= self.__params:
                segments.append((current, current + keys_per_segment))
                current += keys_per_segment

            if current < self.__params:
                segments.append((current, self.__params))
            return segments

        params = 0
        total = 0
        layers_range = []
        for i, size in enumerate(self.__sizes):
            if i not in exclusions:
                layers_range.append((total, total + size))
                params += size
            total += size

        merged = []
        for start, end in layers_range:
            if not merged or merged[-1][1] != start:
                merged.append([start, end])
            else:
                merged[-1][1] = end

        available_ranges = [tuple(r) for r in merged]
        keys_per_segment = int(np.ceil(params / num))

        for r_start, r_end in available_ranges:
            current = r_start
            while current + keys_per_segment <= r_end:
                segments.append((current, current + keys_per_segment))
                current += keys_per_segment
            if current < r_end:
                segments.append((current, r_end))
        return segments
