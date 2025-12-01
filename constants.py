import sys
from src.peer_id import PeerID

ID = PeerID("localhost", int(sys.argv[1]))
NEIGHBORS = [
    PeerID("localhost", 8001),
    PeerID("localhost", 8002),
    PeerID("localhost", 8003),
    PeerID("localhost", 8004),
    PeerID("localhost", 8005),
    PeerID("localhost", 8006),
    PeerID("localhost", 8007),
    PeerID("localhost", 8008),
    PeerID("localhost", 8009),
    PeerID("localhost", 8010),
    PeerID("localhost", 8011),
    PeerID("localhost", 8012),
]

# DATASETS
DATASET = sys.argv[2].split("_")
TRAIN = (
    f"../datasets/{DATASET[0]}/X_train_{DATASET[1]}.csv",
    f"../datasets/{DATASET[0]}/y_train_{DATASET[1]}.csv",
)
VAL = (
    f"../datasets/{DATASET[0]}/X_val.csv",
    f"../datasets/{DATASET[0]}/y_val.csv",
)
TEST = (
    f"../datasets/SERVER/X_test.csv",
    f"../datasets/SERVER/y_test.csv",
)

# TRAIN PARAMS
ROUNDS = 10
EPOCHS = 10
BATCH_SIZE = 32

# SEGMENTS
S = 40
R = 2
