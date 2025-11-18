import pickle
from queue import Queue
import socket
import struct
import threading
import time
from typing import Callable
from src.peer_id import PeerID

RETRY_INTERVAL = 60

class NetworkHandler:
    def __init__(
            self,
            id: PeerID,
            neighbors: list[PeerID] = [],
            on_message: Callable[[PeerID, dict], None] | None = None
        ):
        self.id: PeerID = id
        self.neighbors: list[PeerID] = neighbors
        self.on_message: Callable[[PeerID, dict], None] | None = on_message
        # Available connections
        self.active_connections: dict[PeerID, socket.socket]= {}
        self.lock = threading.Lock()
        # Messages queue
        self.msg_queue = Queue()
        # To close threads
        self.stop_event = threading.Event()

        # Threads
        self.server_thread = None
        self.connection_thread = None
        self.training_thread = None

    def __add_connection(self, peer_id: PeerID, conn: socket.socket):
        self.active_connections[peer_id] = conn
        self.send(peer_id, {
            "type": "handshake",
            "host": self.id.host,
            "port": self.id.port,
        })
        print(f"\033[32m[CONNECTED]\033[0m      {peer_id}")

    def __remove_connection(self, conn: socket.socket):
        with self.lock:
            for pid, c in list(self.active_connections.items()):
                if c == conn:
                    print(f"\033[31m[DISCONNECTED]\033[0m   {pid}")
                    self.active_connections.pop(pid, None)
                    break
        try:
            conn.close()
        except:
            pass

    def __send(self, conn: socket.socket, data: bytes):
        try:
            conn.sendall(struct.pack("!I", len(data)) + data)
        except Exception:
            self.__remove_connection(conn)

    def send(self, peer_id: PeerID, msg: dict):
        with self.lock:
            conn = self.active_connections.get(peer_id)

        if not conn:
            return

        try:
            data = pickle.dumps(msg)
            self.__send(conn, data)
        except Exception:
            print(f"\033[37m[ERROR]\033[0m          Failed to send message to {peer_id}")
            self.__remove_connection(conn)

    def broadcast(self, msg: dict):
        with self.lock:
            peers = list(self.active_connections.keys())

        for pid in peers:
            self.send(pid, msg)

    def __read(self, conn: socket.socket, n: int):
        buf = b""

        while len(buf) < n:
            try:
                chunk = conn.recv(n - len(buf))
            except:
                return None
            
            if not chunk:
                return None
            
            buf += chunk
        return buf

    def __receive_loop(self, conn: socket.socket):
        while not self.stop_event.is_set():
            size_data = self.__read(conn, 4)

            if not size_data:
                break

            msg_size = struct.unpack("!I", size_data)[0]
            data = self.__read(conn, msg_size)

            if not data:
                break

            try:
                msg = pickle.loads(data)
            except:
                continue

            sender_id = None

            with self.lock:
                for pid, c in self.active_connections.items():
                    if c == conn:
                        sender_id = pid
                        break
                    
            self.msg_queue.put((sender_id, conn, msg))
        self.__remove_connection(conn)

    def __dispatch_loop(self):
        while not self.stop_event.is_set():
            sender, conn, msg = self.msg_queue.get()

            if msg.get("type") == "handshake":
                peer_id = PeerID(msg["host"], msg["port"])

                with self.lock:
                    connected = self.active_connections.get(peer_id)

                if not connected:
                    self.__add_connection(peer_id, conn)
                continue
            if self.on_message and sender is not None:
                try:
                    self.on_message(sender, msg)
                except Exception as e:
                    print(f"\033[37m[ERROR]\033[0m          Error in message handler: {e}")

    def __start_server(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(self.id.address)
        server.listen()

        print(
            f"\033[37m[INFO]\033[0m           Running on {self.id}"
        ) 
        try:
            while not self.stop_event.is_set():
                conn, addr = server.accept()
                threading.Thread(target=self.__receive_loop, args=(conn,), daemon=True).start()
        finally:
            server.close()
    
    def __connect_to_peers(self):
        while not self.stop_event.is_set():
            for pid in self.neighbors:
                if pid == self.id:
                    continue

                with self.lock:
                    if pid in self.active_connections:
                        continue

                try:
                    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    
                    client.connect(pid.address)
                    self.__add_connection(pid, client)
                    threading.Thread(target=self.__receive_loop, args=(client,), daemon=True).start()
                except:
                    pass
            time.sleep(RETRY_INTERVAL)

    def start(self):
        self.messages_thread = threading.Thread(target=self.__dispatch_loop, daemon=True)
        self.server_thread = threading.Thread(target=self.__start_server, daemon=True)
        self.connection_thread = threading.Thread(target=self.__connect_to_peers, daemon=True)
        
        self.stop_event.clear()
        self.messages_thread.start()
        self.server_thread.start()
        self.connection_thread.start()
        return self.messages_thread, self.server_thread, self.connection_thread

    def stop(self):
        self.stop_event.set()
        with self.lock:
            for c in self.active_connections.values():
                try:
                    c.close()
                except:
                    pass
            self.active_connections.clear()
        if self.messages_thread:
            self.messages_thread.join(timeout=1)
        if self.server_thread:
            self.server_thread.join(timeout=1)
        if self.connection_thread:
            self.connection_thread.join(timeout=1)
        print(f"\033[37m[INFO]\033[0m           Stop listening on {self.id}")