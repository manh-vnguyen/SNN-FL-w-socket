import socket
import threading
import msgpack
import msgpack_numpy
import json
import torch
import cifar10
import multiprocessing
import time

from collections import OrderedDict
from typing import Dict, List, Tuple
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PeripheralFL():
    def __init__(self) -> None:
        node_id = 0
        self.trainloader, self.testloader = cifar10.load_client_data(node_id)

        self.local_model_parameters = None
        self.local_model_params_payload = None
        self.parent_conn, self.child_conn = None, None
        
        self.local_traing_process = None

        self.current_training_epoch = None
        self.client_logs = []

    def get_parameters(self, model, config: Dict[str, str]) -> List[np.ndarray]:
        return [val.cpu().numpy() for _, val in model.state_dict().items()]


    def set_parameters(self, model, parameters: List[np.ndarray]):
        params_dict = zip(self.local_model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v.astype(float)) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)


    def fit(
        self, conn, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict]:
        # Set model parameters, train model, return updated model parameters
        model = cifar10.load_model().to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0)

        self.set_parameters(parameters)
        cifar10.train(model, optimizer, self.trainloader, DEVICE, 1)
        loss, accuracy = cifar10.test(model, self.testloader, DEVICE)

        conn.send((self.get_parameters(config={}), loss, accuracy))

    def is_running_local_training(self):
        return self.local_traing_process is not None and self.local_traing_process.is_alive()
    
    def discard_local_training(self):
        if self.is_running_local_training():
            self.local_traing_process.terminate()
            self.client_logs.append({
                'epoch': self.current_training_epoch,
                'status': 'TRAINING_TERMINATED'
            })

        del self.local_traing_process, self.parent_conn, self.child_conn
        del self.local_model_parameters, self.local_model_params_payload 

    
    def spawn_new_local_training(self, epoch, parameters):
        self.discard_local_training()
        
        self.parent_conn, self.child_conn = multiprocessing.Pipe()
        self.local_traing_process = multiprocessing.Process(target=self.fit, args=(self.child_conn, parameters, {}))

        self.current_training_epoch = epoch
        self.local_traing_process.start()

    def handle_epoch_completed(self):
        if not self.parent_conn.poll():
            raise Exception("No training result found in Pipe")
        
        # If there are training result, process this result
        fitted_parameters, loss, accuracy = self.parent_conn.recv()

        self.local_model_params = fitted_parameters
        self.local_model_params_payload = msgpack.packb(self.local_model_parameters, default=msgpack_numpy.encode)
        
        self.client_logs.append({
            'epoch': self.current_training_epoch,
            'status': 'TRAINING_COMPLETED',
            'loss': loss,
            'accuracy': accuracy, 
            'params_size': len(self.local_model_params_payload),
            'params_hash': hash(self.local_model_params_payload),
            # # Keep the logs light, we can record the params data to a files later if needed
            # 'params': self.local_model_params,
            # 'params_payload': self.local_model_params_payload
        })
        return self.client_logs[-1]

class Client:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_uid = None
        self.break_connection = False
        self.socket_lock = threading.Lock()

        self.peripheral_fl = PeripheralFL()
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
    
    def send_data(self, data, data_size):
        pass


    def send_to_server(self, message):
        print(f"Sending to server: {message}")
        with self.socket_lock:
            self.sock.sendall(message.encode('utf-8'))

    def start_new_training_epoch(self, server_status):
        global_model_params_size = server_status['global_model_size']
        global_training_epoch = server_status['global_training_epoch']

        # Let server know client is ready for model params transfer
        self.send_to_server(f"CLIENT_READY_TO_RECEIVE_MODEL:::{global_model_params_size}")

        # Receive the model params data and unpack it
        global_model_params = msgpack.unpackb(self.receive_data(global_model_params_size), object_hook=msgpack_numpy.decode)

        # Notify server about received model params
        self.send_to_server(f"CLIENT_CONFIRM_MODEL_RECEIVED:::{global_model_params_size}")

        self.peripheral_fl.spawn_new_local_training(global_training_epoch, global_model_params)

    def local_params_transfer(self, epoch_log):
        self.send_to_server(f"CLIENT_INITIATE_LOCAL_PARAMS_TRANSFER:::{json.dumps(epoch_log['params_size'])}")

        # Try a separate listen from the listen_to_server while loop
        message = self.sock.recv(1024).decode('utf-8')

    def handle_server_send_status(self, message):
        server_status = json.loads(message.split(':::')[1])

        if server_status['aggregation_running']:
            self.peripheral_fl.discard_local_training()
            client_status = {
                'epoch': self.peripheral_fl.current_training_epoch,
                'status': 'AWAIT_NEW_GLOBAL_MODEL'
            }
            self.send_to_server(f"CLIENT_UPDATE_STATUS:::{json.dumps(client_status)}")
        else:
            if server_status['global_training_epoch'] != self.peripheral_fl.current_training_epoch:
                self.start_new_training_epoch(server_status)
            else:
                if self.peripheral_fl.is_running_local_training():
                    client_status = {
                        'epoch': self.peripheral_fl.current_training_epoch,
                        'status': 'TRAINING_IN_PROGRESS'
                    }
                    self.send_to_server(f"CLIENT_UPDATE_STATUS:::{json.dumps(client_status)}")
                else:
                    if server_status['latest_client_result_epoch'] != self.peripheral_fl.current_training_epoch:
                        epoch_log = self.peripheral_fl.handle_epoch_completed()
                        self.local_params_transfer(epoch_log)
                    else:
                        client_status = {
                            'epoch': self.peripheral_fl.current_training_epoch,
                            'status': 'AWAIT_NEW_GLOBAL_MODEL'
                        }
                        self.send_to_server(f"CLIENT_UPDATE_STATUS:::{json.dumps(client_status)}")
        
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
                    elif message.startswith("SERVER_SEND_STATUS:::"):
                        self.handle_server_send_status(message)
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
    PORT = 65432        # The port used by the server
    
    client = Client(HOST, PORT)
    client.connect()
