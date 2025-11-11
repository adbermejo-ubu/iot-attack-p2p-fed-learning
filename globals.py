import sys


def get(lst, index, default=None):
    return lst[index] if 0 <= index < len(lst) else default


ID = get(sys.argv, 1, 0)

# PEER
HOST = "localhost"
PORT = 8000 + int(ID)

# DATASETS
TRAIN = (
    (
        f"datasets/TONIOT/X_train_{ID}.csv",
        f"datasets/TONIOT/y_train_{ID}.csv",
    )
    if ID != None
    else None
)
VAL = ("datasets/TONIOT/X_val.csv", "datasets/TONIOT/y_val.csv")
TEST = ("datasets/TONIOT/X_test.csv", "datasets/TONIOT/y_test.csv")

# PARAMS
BATCH_SIZE = 32
ROUNDS = 10
EPOCHS = 20
