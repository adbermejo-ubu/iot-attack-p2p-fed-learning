import constants
from src.peer import Peer
from src.utils import load_datasets, load_model

if __name__ == "__main__":
    datasets = load_datasets(constants.TRAIN, constants.VAL, constants.TEST)
    model = load_model(datasets.X_test.shape[1])
    peer = Peer(constants.ID, model, datasets, constants.NEIGHBORS)

    try:
        peer.start()
    except KeyboardInterrupt:
        peer.stop()
