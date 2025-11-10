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
from src.model import fed_avg, flat_model, reshape_model

S = 20
R = 2


class Peer:
    round = 0
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
        self.model_inbox = {}
        self.model_inbox_count = 0
        self.train = train_dataset
        self.val = val_dataset
        self.test = test_dataset
        self.neighbors = neighbors
        self.active_connections = {}

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
        buffer = io.BytesIO()

        pickle.dump(msg, buffer)
        buffer = buffer.getvalue()
        size = struct.pack("!I", len(buffer))
        conn.sendall(size + buffer)

    def __receive_msg(self, conn: socket.socket, msg: Any):
        match msg["type"]:
            case "handshake":
                peer_id = (msg["host"], msg["port"])
                if peer_id != self.id and peer_id not in self.active_connections:
                    print(
                        f"\033[32m[CONNECTED]\033[0m      Peer {peer_id[0]}:{peer_id[1]}"
                    )
                    self.active_connections[peer_id] = conn
                    self.__send_msg(
                        conn,
                        {"type": "handshake", "host": self.id[0], "port": self.id[1]},
                    )
            case "request":
                with self.model_lock:
                    # TODO
                    for peer_id, c in list(self.active_connections.items()):
                        if c == conn:
                            print(f"🛑 Mandando el esgmento for segment {msg['start']}-{msg['end']}, al peer {peer_id}")
                            break

                    range_weights = (msg["start"], msg["end"])
                    samples = self.train[0].shape[0]
                    weights = flat_model(self.model)

                    self.__send_msg(
                        conn,
                        {
                            "type": "model",
                            "weights": weights[range_weights[0] : range_weights[1]],
                            "samples": samples,
                            "start": range_weights[0],
                            "end": range_weights[1],
                        },
                    )
            case "model":
                range_weights = (msg["start"], msg["end"])
                samples = msg["samples"]
                weights = np.asarray(msg["weights"], dtype=np.float32)

                for peer_id, c in list(self.active_connections.items()):
                    if c == conn:
                        # TODO
                        print(f"✅ Segmento recibido segment {msg['start']}-{msg['end']} del peer {peer_id}")
                        # print(
                        #     f"                 New segment from peer {peer_id[0]}:{peer_id[1]} in range {msg['start']}:{msg['end']}"
                        # )
                        break
                with self.model_inbox_cond:
                    if range_weights in self.model_inbox:
                        self.model_inbox[range_weights].append((samples, weights))
                    else:
                        self.model_inbox[range_weights] = [(samples, weights)]
                    self.model_inbox_count += 1
                    self.model_inbox_cond.notify_all()

    def __receive_loop(self, conn):
        def receive(conn, n):
            buf = b""

            while len(buf) < n:
                chunk = conn.recv(n - len(buf))

                if not chunk:
                    return None
                buf += chunk
            return buf

        while not self.stop_event.is_set():
            size_data = receive(conn, 4)

            if not size_data:
                break

            msg_size = struct.unpack("!I", size_data)[0]
            data = receive(conn, msg_size)

            if not data:
                break

            msg = pickle.loads(data)

            self.__receive_msg(conn, msg)

    def __start_server(self):
        def handle(conn: socket.socket):
            try:
                self.__receive_loop(conn)
            finally:
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
        return history

    def __request_model(self):
        availables = list(self.active_connections.values())
        print(f"\033[35m[REQUEST MODELS]\033[0m Round {self.round} - {S} segments and {min(R, len(availables))} replicas")
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
                # TODO
                for peer_id, c in list(self.active_connections.items()):
                    if c == peer:
                        print(f"⏳ Esperando el segmento {start}-{end}, al peer {peer_id}")
                        break

        print(f"\033[35m[WAITING MODELS]\033[0m Round {self.round}")
        with self.model_inbox_cond:
            stop = time.time() + 60
            while self.model_inbox_count < min(R, len(availables)) * S and time.time() < stop:
                remaining = stop - time.time()
                if remaining <= 0:
                    break
                self.model_inbox_cond.wait(timeout=remaining)

    def __aggregate(self):
        with self.model_lock:
            print(f"\033[35m[AGGREGATION]\033[0m    Round {self.round}")
            weights = flat_model(self.model)

            with self.model_inbox_cond:
                for start, end in list(self.model_inbox.keys()):
                    local_segment = (self.train[0].shape[0], weights[start:end])
                    segments = self.model_inbox[(start, end)]
                    weights[start:end] = fed_avg([local_segment, *segments])
                model_weights = reshape_model(
                    weights,
                    self.original_shapes,
                    self.original_sizes,
                )
                self.model.set_weights(model_weights)

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
            y_pred = tf.cast(y_pred_probs > 0.5, tf.int8)
            y_true = self.test[1].numpy()
            y_pred = y_pred.numpy()

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
        time.sleep(5)
        while self.round < globals.ROUNDS:
            self.round += 1
            self.__train()
            self.__request_model()
            self.__aggregate()
            self.__evaluate()
        self.stop_event.set()

    # Utilities ###################################################################################
    def init(self):
        try:
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
