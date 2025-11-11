import io
import pickle
import random
import socket
import struct
import threading
import time
from typing import Any
import numpy as np
import tensorflow as tf
from queue import Queue
from keras import Model, callbacks
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import globals
from src.model import fed_avg, fed_avg_sim, flat_model, reshape_model

S = 20
R = 2


class Peer:
    stop_event = threading.Event()
    model_lock = threading.Lock()
    model_inbox_cond = threading.Condition()

    def __init__(
        self,
        id: tuple[str, int],
        model: Model,
        train_dataset,
        val_dataset,
        test_dataset,
        neighbors: list[tuple[str, int]] = [],
    ):
        self.id = id
        self.model = model
        self.train = train_dataset
        self.val = val_dataset
        self.test = test_dataset
        self.neighbors = neighbors

        self.active_connections = {}
        self.msg_queue = Queue()
        self.model_inbox = {}
        self.model_inbox_count = 0
        self.round = 0

        # Get constant data
        self.original_shapes = [w.shape for w in self.model.get_weights()]
        self.original_sizes = [w.size for w in self.model.get_weights()]

        # Show info
        self.model.summary()
        print("\n\033[1mDataset\033[0m")
        print(
            f" \033[1mTrain:\033[0m {self.train[0].shape} {self.train[1].shape}\n\n"
        )
        print(
            f" \033[1mValidation:\033[0m {self.val[0].shape} {self.val[1].shape}\n\n"
        )
        print(f" \033[1mTest:\033[0m {self.test[0].shape} {self.test[1].shape}\n\n")

    # Communication ###############################################################################
    def __send_msg(self, conn: socket.socket, msg: Any):
        data = pickle.dumps(msg)
        conn.sendall(struct.pack("!I", len(data)) + data)

    def __register_peer(self, conn: socket.socket, peer_id: tuple[str, int]):
        print(f"\033[32m[CONNECTED]\033[0m    Peer {peer_id[0]}:{peer_id[1]}")
        self.active_connections[peer_id] = conn

    def __handle_message(self, conn: socket.socket, msg: Any):
        mtype = msg["type"]

        if mtype == "handshake":
            peer_id = (msg["host"], msg["port"])
            if peer_id != self.id and peer_id not in self.active_connections:
                self.__register_peer(conn, peer_id)
                self.__send_msg(conn, {"type": "handshake", "host": self.id[0], "port": self.id[1]})
        elif mtype == "request":
            with self.model_lock:
                weights = self.cached_weights
            start, end = msg["start"], msg["end"]
            samples = self.train[0].shape[0]
            self.__send_msg(conn, {
                "type": "model",
                "weights": weights[start:end],
                "samples": samples,
                "start": start,
                "end": end,
            })
        elif mtype == "model":
            start, end = msg["start"], msg["end"]
            samples = msg["samples"]
            weights = np.asarray(msg["weights"], dtype=np.float32)

            # identify sender
            for peer_id, c in self.active_connections.items():
                if c == conn:
                    print(f"    Received [{start}:{end}] segment from {peer_id[0]}:{peer_id[1]}")
                    break

            with self.model_inbox_cond:
                self.model_inbox.setdefault((start, end), []).append((samples, weights))
                self.model_inbox_count += 1
                self.model_inbox_cond.notify_all()

    def __dispatch_messages(self):
        while not self.stop_event.is_set():
            conn, msg = self.msg_queue.get()
            try:
                self.__handle_message(conn, msg)
            except Exception as e:
                print("[ERROR] dispatch:", e)

    def __receive_loop(self, conn: socket.socket):
        def read(n):
            buf = b""

            while len(buf) < n:
                chunk = conn.recv(n - len(buf))

                if not chunk:
                    return None
                buf += chunk
            return buf

        while not self.stop_event.is_set():
            size_data = read(4)

            if not size_data:
                break

            msg_size = struct.unpack("!I", size_data)[0]
            data = read(msg_size)
            
            if not data:
                break
            
            msg = pickle.loads(data)
            
            self.msg_queue.put((conn, msg))

    def __start_server(self):
        def handle(conn: socket.socket):
            try:
                self.__receive_loop(conn)
            finally:
                # remove peer on disconnect
                for peer_id, c in list(self.active_connections.items()):
                    if c == conn:
                        print(
                            f"\033[31m[DISCONNECTED]\033[0m   Peer {peer_id[0]}:{peer_id[1]}"
                        )
                        self.active_connections.pop(peer_id, None)
                        break
                conn.close()

        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(self.id)
        server.listen()
        print(
            f"\033[37m[INFO]\033[0m           Peer running on {self.id[0]}:{self.id[1]}"
        )

        try:
            while not self.stop_event.is_set():
                conn, _ = server.accept()
                threading.Thread(target=handle, args=(conn,), daemon=True).start()
        finally:
            server.close()

    def __connect_to_peers(self):
        def handle(conn: socket.socket, peer_id: tuple[str, int]):
            try:
                self.__receive_loop(conn)
            finally:
                print(f"\033[31m[DISCONNECTED]\033[0m   Peer {peer_id[0]}:{peer_id[1]}")
                self.active_connections.pop(peer_id, None)
                conn.close()

        while not self.stop_event.is_set():
            for peer_id in self.neighbors:
                try:
                    if peer_id == self.id or peer_id in self.active_connections:
                        continue
                    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    client.connect(peer_id)
                    self.__send_msg(
                        client,
                        {"type": "handshake", "host": self.id[0], "port": self.id[1]},
                    )
                    threading.Thread(
                        target=handle, args=(client, peer_id), daemon=True
                    ).start()
                except:
                    pass
            time.sleep(5)

    # Model Training ##############################################################################
    def __train(self):
        with self.model_lock:
            print(
                f"\033[34m[TRAIN]\033[0m          Round {self.round} - {globals.EPOCHS} epochs"
            )
            history = self.model.fit(
                self.train[0],
                self.train[1],
                batch_size=globals.BATCH_SIZE,
                epochs=globals.EPOCHS,
                verbose=1,
                validation_data=self.val,
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
                    # callbacks.ModelCheckpoint(
                    #     filepath=f"output/{self.id}.keras",
                    #     monitor="val_loss",
                    #     save_best_only=True,
                    #     verbose=1,
                    # ),
                ],
            ).history
        with self.model_lock:
            self.cached_weights = flat_model(self.model)
        return history

    def __request_model(self):
        availables = list(self.active_connections.values())
        replicas = min(R, len(availables))
        expected_segments = replicas * S

        print(f"\033[35m[REQUEST MODELS]\033[0m Round {self.round} - {S} segments and {replicas} replicas")

        with self.model_inbox_cond:
            self.model_inbox.clear()
            self.model_inbox_count = 0

        params_num = sum(self.original_sizes)
        keys_per_segment = int(np.ceil(params_num / S))

        for i in range(0, params_num, keys_per_segment):
            start = i
            end = min(start + keys_per_segment, params_num)
            peers_to_ask = random.sample(availables, min(R, len(availables)))

            for peer in peers_to_ask:
                self.__send_msg(peer, {"type": "request", "start": start, "end": end})

        with self.model_inbox_cond:
            timeout = time.time() + 60
            while self.model_inbox_count < expected_segments and time.time() < timeout:
                self.model_inbox_cond.wait(timeout=timeout - time.time())

    def __aggregate(self):
        print(f"\033[35m[AGGREGATION]\033[0m    Round {self.round}")
        weights = np.copy(self.cached_weights)

        with self.model_inbox_cond:
            for (start, end), segments in self.model_inbox.items():
                local = (self.train[0].shape[0], weights[start:end])
                weights[start:end] = fed_avg_sim(local, segments, 1) # fed_avg([local, *segments])

        with self.model_lock:
            new_weights = reshape_model(weights, self.original_shapes, self.original_sizes)
            self.model.set_weights(new_weights)

    def __evaluate(self):
        with self.model_lock:
            print(f"\033[33m[EVALUATION]\033[0m     Round {self.round}")
            loss = self.model.evaluate(
                self.test[0],
                self.test[1],
                batch_size=globals.BATCH_SIZE,
                return_dict=True,
                verbose=1,
            )["loss"]

            y_pred_probs = self.model.predict(
                self.test[0], batch_size=globals.BATCH_SIZE, verbose=0
            )
            y_pred = tf.cast(y_pred_probs > 0.5, tf.int8).numpy()
            y_true = self.test[1].numpy()

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
        self.save_model()

    def __fed_round(self):
        time.sleep(3)
        while self.round < globals.ROUNDS  and not self.stop_event.is_set():
            self.round += 1
            self.__train()
            self.__request_model()
            self.__aggregate()
            self.__evaluate()
        self.stop_event.set()

    # Utilities ###################################################################################
    def init(self):
        try:
            threading.Thread(target=self.__dispatch_messages, daemon=True).start()
            threading.Thread(target=self.__start_server, daemon=True).start()
            threading.Thread(target=self.__connect_to_peers, daemon=True).start()
            threading.Thread(target=self.__fed_round, daemon=True).start()

            while not self.stop_event.is_set():
                time.sleep(1)
        except KeyboardInterrupt:
            pass

    def save_model(self):
        with self.model_lock:
            self.model.save(f"output/model-{globals.ID}.keras")
