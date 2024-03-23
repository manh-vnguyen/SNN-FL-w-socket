import socket
import threading
import msgpack

class Client:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_uid = None
        self.break_connection = False
        # self.sock.settimeout(5.0)

    def connect(self):
        try:
            self.sock.connect((self.host, self.port))
            print(f"Connected to server {self.host}:{self.port}")
        except Exception as e:
            print(f"Failed to connect to server {self.host}:{self.port}: {e}")
            return

        self.listen_to_server()
        self.close()

    def handle_identity_prompt(self):
        if self.client_uid is None:
            self.send_to_server("NEW_CLIENT_REQUEST_ID")
        else:
            self.send_to_server(f"EXISTING_CLIENT_SUBMIT_ID:{self.client_uid}")

    def handle_new_identity_assigned(self, message):
        if self.client_uid is not None:
            print(f"Server attempts to assign new id on existing client: {self.client_uid}")
            self.break_connection = True
        else:
            self.client_uid = message.split(':')[1]
            print(f"Server assigned client_uid: {self.client_uid}. Connection established.")
            self.send_to_server(f"NEW_CLIENT_SUBMIT_ACK:{self.client_uid}")

    def handle_existing_client_ack(self, message):
        if self.client_uid != message.split(':')[1]:
            print(f"Server acknowledged wrong client id")
            self.break_connection = True
        else:
            print(f"Connection with server established")

    def handle_unrecognized_message(self, message):
        print(f"Unable to recognize message: {message}")
        # self.sock.sendall(f"Unable to recognize message: {message}".encode('utf-8'))

    def receive_data(self, data_size):
        received_data = b''
        while len(received_data) < data_size:
            try:
                more_data = self.sock.recv(data_size - len(received_data))
                if not more_data:
                    raise Exception("Server closed the connection unexpectedly.")
                received_data += more_data
            except socket.timeout:
                raise Exception("Timed out waiting for data from server.")
            
        return received_data

    def stop_current_training(self):
        pass
    
    def spawn_new_local_training(self, model_params):
        pass

    def send_to_server(self, message):
        self.sock.sendall(message.encode('utf-8'))

    def handle_newly_updated_model(self, message):
        self.stop_current_training()

        model_params_size = int(message.split(':')[1])
        self.send_to_server(f"CLIENT_READY_TO_RECEIVE")
        model_params = msgpack.unpackb(self.receive_data(model_params_size))
        self.send_to_server(f"CLIENT_ACKNOWLEDGE_MODEL:{hash(model_params)}")
        self.spawn_new_local_training(model_params)

    def listen_to_server(self):
        while not self.break_connection:
            try:
                message = self.sock.recv(1024).decode('utf-8')
                if message:
                    print(f"Received from server: {message}")
                    if message == "SERVER_IDENTIFY_CLIENT":
                        self.handle_identity_prompt()
                    elif message.startswith("SERVER_ASSIGN_NEW_CLIENT_ID"):
                        self.handle_new_identity_assigned(message)
                    elif message.startswith("SERVER_ACK_EXISTING_CLIENT"):
                        self.handle_existing_client_ack(message)
                    elif message.startswith("SERVER_NOTIFY_UPDATED_MODEL"):
                        self.handle_newly_updated_model(message)
                    else:
                        self.handle_unrecognized_message(message)

                else:
                    print("Connection closed by the server.")
                    break
            except Exception as e:
                print(f"Error receiving data: {e}")
                break

    def close(self):
        self.sock.close()
        print("Connection closed.")

if __name__ == "__main__":
    HOST = '127.0.0.1'  # The server's hostname or IP address
    PORT = 65431        # The port used by the server
    
    client = Client(HOST, PORT)
    client.connect()
