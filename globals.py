import sys
from src.peer_id import PeerID

DATASET_NAME = str(sys.argv[2]).upper()
ID = PeerID("localhost", 8000 + int(sys.argv[1]) + (0 if DATASET_NAME == "TONIOT" else 4))

# DATASETS
TRAIN = (
    (
        f"datasets/{DATASET_NAME}/X_train_{sys.argv[1]}.csv",
        f"datasets/{DATASET_NAME}/y_train_{sys.argv[1]}.csv",
    )
    if ID != None
    else None
)
VAL = (f"datasets/{DATASET_NAME}/X_val.csv", f"datasets/{DATASET_NAME}/y_val.csv")
TEST = ("datasets/SERVER/X_test.csv", "datasets/SERVER/y_test.csv")

# PARAMS
BATCH_SIZE = 256
ROUNDS = 10
EPOCHS = 10
R = 4
S = 40
