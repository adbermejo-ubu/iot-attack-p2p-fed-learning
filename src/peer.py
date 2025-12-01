import os
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
import constants
from src.aggregation import fed_avg, fed_avg_sim
from src.datasets import Datasets
from src.model import P2PModel
from src.network import NetworkHandler
from src.peer_id import PeerID

TIMEOUT = 60 * 2


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
        self.expected_models = {}
        self.expected_models_cond = threading.Condition()
        self.models_inbox = {}
        self.models_inbox_count = 0
        self.models_inbox_cond = threading.Condition()

        # Federated state
        self.round = 0
        self.max_rounds = constants.ROUNDS

        # Local state
        self.max_epochs = constants.EPOCHS
        self.batch_size = constants.BATCH_SIZE
        self.model_lock = threading.Lock()
        with self.model_lock:
            self.cached_weights = self.model.flat()
            self.cached_weights_round = self.round

        # Thread control
        self.stop_event = threading.Event()
        self.training_thread: threading.Thread | None = None
        self.training_done = threading.Event()

        # Build output directory
        os.makedirs("output", exist_ok=True)

    # Reset state
    def __reset_inbox(self):
        with self.models_inbox_cond:
            old = self.models_inbox.copy()

            self.models_inbox.clear()
            self.models_inbox_count = 0
            return old

    def __reset_expected(self):
        with self.expected_models_cond:
            old = self.expected_models.copy()

            self.expected_models.clear()
            return old

    def __reset_state(self):
        self.__reset_inbox()
        self.__reset_expected()

    # Message callback
    def __on_message(self, sender: PeerID, msg: dict):
        mtype = msg.get("type")

        if mtype == "request":
            round = msg["round"]
            start, end = msg["start"], msg["end"]

            if round > self.cached_weights_round:
                self.network_handler.send(
                    sender,
                    {"type": "model", "not_ready": True, "start": start, "end": end},
                )
            else:
                weights = np.copy(self.cached_weights[start:end])
                samples = self.datasets.train_size
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
            not_ready = msg.get("not_ready", False)
            start, end = msg["start"], msg["end"]

            if not_ready:
                with self.expected_models_cond:
                    self.expected_models.setdefault((start, end), set()).add(sender)
                    self.expected_models_cond.notify_all()
                return

            weights = np.asarray(msg["weights"], dtype=np.float32)
            samples = msg["samples"]

            print(f"                 Received [{start}:{end}] segment from {sender}")
            with self.expected_models_cond:
                self.expected_models.get((start, end), set()).discard(sender)
                self.expected_models_cond.notify_all()
            with self.models_inbox_cond:
                self.models_inbox.setdefault((start, end), []).append(
                    (samples, weights)
                )
                self.models_inbox_count += 1
                self.models_inbox_cond.notify_all()
        else:
            print(
                f"\033[37m[WARN]\033[0m           Unknown message type from {sender}: {mtype}"
            )

    # Federated loop
    def __federated_loop(self):
        try:
            while not self.stop_event.is_set() and (self.round < self.max_rounds):
                self.round += 1
                # Reset the training state
                self.__reset_state()
                # Train the model
                self.__train()
                # Federated learning
                self.__federate()
                # Evaluate the model
                self.__evaluate()
        finally:
            self.training_done.set()

    def __train(self):
        with self.model_lock:
            print(
                f"\033[34m[TRAIN]\033[0m          Round {self.round}/{self.max_rounds} - {self.max_epochs} epochs"
            )
            history = self.model.fit(
                self.datasets.X_train,
                self.datasets.y_train,
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
            self.cached_weights_round = self.round
        return history

    def __federate(self):
        # Request segments to the clients
        def request_models():
            with self.network_handler.lock:
                availables = list(self.network_handler.active_connections.keys())

            if not availables:
                print(
                    f"\033[37m[WARN]\033[0m           No peers available to request models from"
                )
                return

            replicas = min(constants.R, len(availables))
            print(
                f"\033[35m[REQUEST MODELS]\033[0m Round {self.round}/{self.max_rounds} - {constants.S} segments and {replicas} replicas"
            )

            segments = self.model.segment(constants.S)
            for start, end in segments:
                peers_to_ask = random.sample(availables, replicas)

                for pid in peers_to_ask:
                    self.network_handler.send(
                        pid,
                        {
                            "type": "request",
                            "round": self.round,
                            "start": start,
                            "end": end,
                        },
                    )
            return replicas * constants.S  # Expected segments

        # Retry requests
        def retry_request(min_segments: int):
            deadline = time.time() + TIMEOUT

            with self.models_inbox_cond:
                while True:
                    # We have all the models
                    if self.models_inbox_count >= min_segments:
                        break

                    # Timeout is consumed
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        print(
                            f"\033[33m[TIMEOUT]\033[0m        Exceeded wait time of {TIMEOUT}s"
                        )
                        break

                    # There are not models for retry
                    pending_retries = self.__reset_expected()

                    for (start, end), peers in pending_retries.items():
                        for pid in peers:
                            self.network_handler.send(
                                pid,
                                {
                                    "type": "request",
                                    "round": self.round,
                                    "start": start,
                                    "end": end,
                                },
                            )
                    self.models_inbox_cond.wait(timeout=min(remaining, 5))

        # Aggregate segments
        def aggregate_models():
            print(
                f"\033[35m[AGGREGATION]\033[0m    Round {self.round}/{self.max_rounds}"
            )

            inbox = self.__reset_inbox()
            if not inbox:
                print(
                    "\033[33m[AGGREGATION]\033[0m    No segments received, skipping aggregation."
                )
                return

            weights = np.copy(self.cached_weights)
            for (start, end), segments in inbox.items():
                local = (self.datasets.train_size, weights[start:end])
                weights[start:end] = fed_avg([local, *segments])

            with self.model_lock:
                self.model.reconstruct(weights)
                self.cached_weights = weights

        expected_segments = request_models()
        retry_request(expected_segments)
        aggregate_models()

    def __evaluate(self):
        with self.model_lock:
            print(
                f"\033[33m[EVALUATION]\033[0m     Round {self.round}/{self.max_rounds}"
            )
            loss = self.model.evaluate(
                self.datasets.X_test,
                self.datasets.y_test,
                batch_size=self.batch_size,
                return_dict=True,
                verbose=1,
            )["loss"]
            y_pred_probs = self.model.predict(
                self.datasets.X_test, batch_size=self.batch_size, verbose=0
            )
            y_pred = tf.cast(y_pred_probs > 0.5, tf.int8).numpy()
            y_true = self.datasets.y_test.numpy()

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        auc = (
            roc_auc_score(y_true, y_pred_probs) if len(np.unique(y_true)) > 1 else None
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
            self.model.save(f"output/{str(self.id).replace(':', '_')}.keras")

    # Control
    def start(self):
        self.stop_event.clear()
        self.network_handler.start()

        # Start training
        self.training_thread = threading.Thread(
            target=self.__federated_loop,
            daemon=True,
        )
        self.training_thread.start()
        # End the training
        self.training_done.wait()
        print("\033[37m[INFO]\033[0m           Training done")
        time.sleep(TIMEOUT * 4)
        self.stop()

    def stop(self):
        self.stop_event.set()
        self.network_handler.stop()
        if self.training_thread and self.training_thread.is_alive():
            self.training_thread.join(timeout=5)
