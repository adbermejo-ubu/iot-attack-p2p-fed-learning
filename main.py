import globals
from src.model import load_model
from src.datasets import load_datasets
from src.peer import Peer


if __name__ == "__main__":
    train_dataset, val_dataset, test_dataset = load_datasets(globals.TRAIN, globals.VAL, globals.TEST)
    model = load_model(test_dataset[0].shape[1])

    Peer(
        (globals.HOST, globals.PORT),
        model,
        train_dataset,
        val_dataset,
        test_dataset,
        [
            ("localhost", 8001),
            ("localhost", 8002),
            ("localhost", 8003),
            ("localhost", 8004),
        ],
    ).init()
