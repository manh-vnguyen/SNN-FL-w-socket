import socket
import threading
import uuid
import msgpack
import multiprocessing
import time
import copy


class Server:
    def __init__(self, host='127.0.0.1', port=65431):
        self.host = host
        self.port = port
        self.clients = {}  # Maps UIDs to client sockets
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))

        self.lock = threading.Lock()
        self.processes = []

        self.current_training_epoch = 0
        self.client_model_record = {}
        self.client_model_logs = {}

        self.min_fit_clients = 5

    def identify_client(self, client_socket):
        client_socket.send(f"SERVER_IDENTIFY_CLIENT".encode('utf-8'))

    def create_new_client(self, client_socket):
        client_uid = str(uuid.uuid4())
        self.clients[client_uid] = client_socket
        
        client_socket.send(f"SERVER_ASSIGN_NEW_CLIENT_ID:{client_uid}".encode('utf-8'))

        return client_uid

    def update_existing_client(self, client_socket, message):
        client_uid = message.split(':')[1]

        if client_uid in self.clients:
            self.clients[client_uid] = client_socket
            client_socket.send(f"SERVER_ACK_EXISTING_CLIENT:{client_uid}".encode('utf-8'))

            return "ACCEPT_EXISTING_CLIENT_UID"
        else:
            print(f"Reject existing client, uid: {client_uid}")
            return "REJECT_EXISTING_CLIENT_UID"

    def print_connection_ack(self, message, new_client=False):
        client_uid = message.split(':')[1]
        if new_client:
            print(f"New client joined the network. Connection established. UID: {client_uid}")
        else:
            print(f"Existing client reconnected. UID: {client_uid}")

    def handle_unrecognized_message(self, client_socket, message):
        print(f"Unable to recognize message: {message}")
        # client_socket.sendall(f"Unable to recognize message: {message}".encode('utf-8'))

    def handle_client_connection(self, client_socket, client_addr):
        # client_socket.settimeout(5.0)
        client_uid = None
        self.identify_client(client_socket)

        while True:
            try:
                print("Check")
                message = client_socket.recv(1024).decode('utf-8')
                if message:
                    print(f"Message from {client_addr}: {message}")
                    
                    if message == "NEW_CLIENT_REQUEST_ID":
                        client_uid = self.create_new_client(client_socket)
                    elif message.startswith("EXISTING_CLIENT_SUBMIT_ID"):
                        if self.update_existing_client(client_socket, message) == "REJECT_EXISTING_CLIENT_UID":
                            break
                        else:
                            client_uid = message.split(':')[1]
                    elif message.startswith("NEW_CLIENT_SUBMIT_ACK"):
                        self.print_connection_ack(message, new_client=True)
                    elif message.startswith("EXISTING_CLIENT_SUBMIT_ACK"):
                        self.print_connection_ack(message, new_client=False)
                    elif message.startswith("CLIENT_NOTIFY_LOCAL_MODEL"):
                        self.handle_client_local_model(message, client_uid)

                    else:
                        self.handle_unrecognized_message(client_socket, message)
                else:
                    break
            except ConnectionResetError:
                break

        print(f"Connection closed. Address: {client_addr}. Client: {client_uid}")
        if client_uid in self.clients:
            self.clients[client_uid] = None
        client_socket.close()

    def send_to_specific_client(self, uid, message):
        if uid in self.clients:
            self.clients[uid].send(message.encode('utf-8'))
            print(f"Sent to {uid}: {message}")
        else:
            print(f"Client UID {uid} not found.")

    def run(self):
        self.server_socket.listen()
        print(f"Server listening on {self.host}:{self.port}")
        print("Server running. Ctrl+C to stop.")
        try:
            while True:
                client_socket, client_addr = self.server_socket.accept()
                thread = threading.Thread(target=self.handle_client_connection, args=(client_socket, client_addr))
                thread.start()
        except KeyboardInterrupt:
            print("Server stopping...")
        finally:
            self.server_socket.close()
            for p in self.processes:
                p.join()  # Wait for all processes to complete

if __name__ == "__main__":
    server = Server()
    server.run()
