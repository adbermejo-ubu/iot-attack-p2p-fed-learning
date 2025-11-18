import random
import threading
import time
import numpy as np
import tensorflow as tf
from keras import callbacks
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import globals
from src.aggregation import fed_avg
from src.datasets import Datasets
from src.model import P2PModel
from src.network import NetworkHandler
from src.peer_id import PeerID

TIMEOUT = 60 * 30


class Peer:
    def __init__(
        self,
        id: PeerID,
        model: P2PModel,
        datasets: Datasets,
        neighbors: list[PeerID] = [],
    ):
        self.id: PeerID = id
        self.model: P2PModel = model
        self.datasets: Datasets = datasets
        self.network_handler: NetworkHandler = NetworkHandler(
            id, neighbors, on_message=self.__on_message
        )

        # Inbox models
        self.models_inbox = {}
        self.models_inbox_count = 0
        self.models_inbox_cond = threading.Condition()

        # Federated state
        self.round = 0
        self.max_rounds = globals.ROUNDS

        # Local state
        self.max_epochs = globals.EPOCHS
        self.batch_size = globals.BATCH_SIZE
        self.model_lock = threading.Lock()
        with self.model_lock:
            self.cached_weights = self.model.flat()

        # Thread control
        self.stop_event = threading.Event()
        self.training_thread: threading.Thread | None = None
        self.training_done = threading.Event()

    def __on_message(self, sender: PeerID, msg: dict):
        mtype = msg.get("type")

        if mtype == "request":
            start, end = msg["start"], msg["end"]

            with self.model_lock:
                weights = np.copy(self.cached_weights[start:end])
            samples = int(self.datasets.train[0].shape[0])
            self.network_handler.send(
                sender,
                {
                    "type": "model",
                    "weights": weights,
                    "samples": samples,
                    "start": start,
                    "end": end,
                },
            )
        elif mtype == "model":
            start, end = msg["start"], msg["end"]
            weights = np.asarray(msg["weights"], dtype=np.float32)
            samples = msg["samples"]

            print(f"                 Received [{start}:{end}] segment from {sender}")
            with self.models_inbox_cond:
                self.models_inbox.setdefault((start, end), []).append((samples, weights))
                self.models_inbox_count += 1
                self.models_inbox_cond.notify_all()
        else:
            print(f"\033[37m[WARN]\033[0m           Unknown message type from {sender}: {mtype}")

    # Training
    def __train(self):
        with self.model_lock:
            print(
                f"\033[34m[TRAIN]\033[0m          Round {self.round}/{self.max_rounds} - {self.max_epochs} epochs"
            )
            history = self.model.fit(
                self.datasets.train[0],
                self.datasets.train[1],
                batch_size=self.batch_size,
                epochs=self.max_epochs,
                verbose=1,
                validation_data=self.datasets.val,
                callbacks=[
                    callbacks.EarlyStopping(
                        monitor="val_loss",
                        patience=15,
                        min_delta=1e-4,
                        restore_best_weights=True,
                        verbose=1,
                    ),
                    callbacks.ReduceLROnPlateau(
                        monitor="val_loss",
                        factor=0.2,
                        patience=7,
                        min_lr=1e-6,
                        verbose=1,
                    ),
                ],
            ).history
            self.cached_weights = self.model.flat()
        return history

    # Share (request segments)
    def __share(self):
        with self.network_handler.lock:
            availables = list(self.network_handler.active_connections.keys())

        if not availables:
            print(f"\033[37m[WARN]\033[0m           No peers available to request models from")
            return

        replicas = min(globals.R, len(availables))
        expected_segments = replicas * globals.S

        print(
            f"\033[35m[REQUEST MODELS]\033[0m Round {self.round}/{self.max_rounds} - {globals.S} segments and {replicas} replicas"
        )

        with self.models_inbox_cond:
            self.models_inbox.clear()
            self.models_inbox_count = 0

        segments = self.model.segment(globals.S)
        for (start, end) in segments:
            peers_to_ask = random.sample(availables, replicas)

            for pid in peers_to_ask:
                self.network_handler.send(pid, {"type": "request", "start": start, "end": end})

        deadline = time.time() + TIMEOUT
        with self.models_inbox_cond:
            while self.models_inbox_count < expected_segments:
                remaining = deadline - time.time()
                if remaining <= 0:
                    print("\033[37m[TIMEOUT]\033[0m        Waiting for model segments timed out")
                    break
                self.models_inbox_cond.wait(remaining)

    # Aggregate
    def __aggregate(self):
        print(f"\033[35m[AGGREGATION]\033[0m    Round {self.round}/{self.max_rounds}")

        with self.models_inbox_cond:
            inbox = {k: list(v) for k, v in self.models_inbox.items()}

        if not inbox:
            print("\033[33m[AGGREGATION]\033[0m    No segments received, skipping aggregation.")
            return

        weights = np.copy(self.cached_weights)
        for (start, end), segments in inbox.items():
            local = (int(self.datasets.train[0].shape[0]), weights[start:end])

            weights[start:end] = fed_avg([local, *segments])

        with self.model_lock:
            self.model.reconstruct(weights)
            self.cached_weights = self.model.flat()

    # Evaluate
    def __evaluate(self):
        with self.model_lock:
            print(f"\033[33m[EVALUATION]\033[0m     Round {self.round}/{self.max_rounds}")
            loss = self.model.evaluate(
                self.datasets.test[0],
                self.datasets.test[1],
                batch_size=self.batch_size,
                return_dict=True,
                verbose=1,
            )["loss"]
            y_pred_probs = self.model.predict(
                self.datasets.test[0], batch_size=self.batch_size, verbose=0
            )
            y_pred = tf.cast(y_pred_probs > 0.5, tf.int8).numpy()
            y_true = self.datasets.test[1].numpy()

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        )
        recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        auc = (
            roc_auc_score(y_true, y_pred_probs)
            if len(np.unique(y_true)) > 1
            else None
        )
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
        print(f"    - accuracy: {accuracy}")
        print(f"    - loss: {loss}")
        print(f"    - precision: {precision}")
        print(f"    - recall: {recall}")
        print(f"    - f1-score: {f1}")
        print(f"    - auc: {auc}")
        print(f"    - confusion-matrix: {cm}")
        with self.model_lock:
            self.model.save(f"output/model-{self.id}.keras")

    # Training loop
    def __training_loop(self):
        try:
            while not self.stop_event.is_set() and (self.round < self.max_rounds):
                self.round += 1
                self.__train()
                self.__share()
                self.__aggregate()
                self.__evaluate()
        finally:
            self.training_done.set()

    # Control
    def start(self):
        self.stop_event.clear()
        self.network_handler.start()

        self.training_thread = threading.Thread(
            target=self.__training_loop,
            daemon=True,
        )
        self.training_thread.start()
        # End the training
        self.training_done.wait()
        self.stop()

    def stop(self):
        self.stop_event.set()
        self.network_handler.stop()
        if self.training_thread and self.training_thread.is_alive():
            self.training_thread.join(timeout=5)
            