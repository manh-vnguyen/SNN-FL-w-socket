import socket
import threading
import msgpack
import msgpack_numpy
import json
import torch
from torch import multiprocessing
import time
import os
import copy
import pickle
import hashlib

from collections import OrderedDict
from typing import Dict, List, Tuple
import numpy as np
import cifar10

CLIENT_DATABASE_PATH = 'client_database.json'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sha256_hash(data):
    hash_object = hashlib.sha256()
    hash_object.update(data)
    
    return hash_object.hexdigest()

class PeripheralFL():
    def __init__(self) -> None:
        self.node_id = 0
        # self.trainloader, self.testloader = cifar10.load_client_data(self.node_id)

        self.local_model_params = None
        self.local_model_payload = None
        self.parent_conn, self.child_conn = None, None
        
        self.local_traing_process = None

        self.current_training_epoch = None
        self.client_logs = []

    def get_parameters(self, model, config: Dict[str, str]) -> List[np.ndarray]:
        return [val.cpu().numpy() for _, val in model.state_dict().items()]


    def set_parameters(self, model, parameters: List[np.ndarray]):
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v.astype(float)) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)


    def fit(
        self, conn, 
        parameters: List[np.ndarray], 
        config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict]:
        
        print("We're training over here")
        # Load data
        trainloader, testloader = cifar10.load_client_data(self.node_id)

        # Set model parameters, train model, return updated model parameters
        model = cifar10.load_model().to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0)

        self.set_parameters(model, parameters)
        cifar10.train(model, optimizer, trainloader, DEVICE, 1)
        loss, accuracy = cifar10.test(model, testloader, DEVICE)
        trained_local_params = self.get_parameters(model, config={})

        with open('/tmp/data/training_result.pkl', 'wb') as file:
            pickle.dump((trained_local_params, loss, accuracy), file)

        conn.send("SUCCESS")
        conn.close()
            

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
        del self.local_model_params, self.local_model_payload

    
    def spawn_new_local_training(self, global_training_epoch, global_model_params):
        self.discard_local_training()
        
        self.parent_conn, self.child_conn = multiprocessing.Pipe()
        self.local_traing_process = multiprocessing.Process(target=self.fit, args=(self.child_conn, global_model_params, {}))

        self.current_training_epoch = global_training_epoch
        self.local_traing_process.start()

    def handle_epoch_completed(self):
        if not self.parent_conn.poll():
            raise Exception("No training result found in Pipe")
        
        self.parent_conn.recv()
        
        # If there are training result, process this result
        with open('/tmp/data/training_result.pkl', 'rb') as file:
            self.local_model_params, loss, accuracy = pickle.load(file)

        self.local_model_payload = msgpack.packb({
                'epoch': self.current_training_epoch,
                'params': self.local_model_params, 
                'accuracy': accuracy,
                'loss': loss,
            }, default=msgpack_numpy.encode)
        
        self.client_logs.append({
            'epoch': self.current_training_epoch,
            'status': 'TRAINING_COMPLETED',
            'loss': loss,
            'accuracy': accuracy, 
            'local_model_payload_size': len(self.local_model_payload),
            'local_model_payload_hash': sha256_hash(self.local_model_payload),
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
        self.socket_lock = threading.Lock()

        self.break_connection = False
        self.database = self.read_or_initiate_database()
        self.client_uid = self.database['client_uid']
        self.connection_established = False

        self.peripheral_fl = PeripheralFL()
        # self.sock.settimeout(5.0)

    def write_database(self, data):
        with open(CLIENT_DATABASE_PATH, 'w') as file:
            json.dump(data, file)

    def read_or_initiate_database(self):
        if not os.path.isfile(CLIENT_DATABASE_PATH):
            self.write_database({
                'client_uid': None
            })

        with open(CLIENT_DATABASE_PATH, 'r') as file:
            data = json.load(file)
        
        return data    

    def connect(self):
        try:
            self.sock.connect((self.host, self.port))
            print(f"Connected to server {self.host}:{self.port}")
        except Exception as e:
            print(f"Failed to connect to server {self.host}:{self.port}: {e}")
            return

        self.listen_to_server()
        self.close()

    def handle_unrecognized_message(self, message):
        print(f"Unable to recognize message: {message}")

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
    
    def receive_global_model(self, global_model_payload_size):
        # Let server know client is ready for model params transfer
        self.send_to_server(f"CLIENT_INITIATE_GLOBAL_MODEL_RECEIVE:::{global_model_payload_size}")

        # Receive the model params data and unpack it
        global_model_payload = self.receive_data(global_model_payload_size)
        global_model_params = msgpack.unpackb(global_model_payload, object_hook=msgpack_numpy.decode)

        # Notify server about received model params
        self.send_to_server(f"CLIENT_CONFIRM_MODEL_RECEIVED:::{sha256_hash(global_model_payload)}")

        return global_model_params

    def local_params_transfer(self, epoch_log):
        self.send_to_server(f"CLIENT_INITIATE_LOCAL_PARAMS_TRANSFER:::{epoch_log['local_model_payload_size']}")

        # Try a separate listen from the listen_to_server while loop
        message = self.sock.recv(1024).decode('utf-8')
        print(message)

        if not message.startswith("SERVER_READY_TO_RECEIVE") or int(message.split(':::')[1]) != epoch_log['local_model_payload_size']:
            raise Exception("Not receiving the right signal confirmation from server for params transfer")
        
        self.transfer_data_to_server(self.peripheral_fl.local_model_payload)

        message = self.sock.recv(1024).decode('utf-8')
        print(f"Receiving from server: {message}")
        print(json.dumps(epoch_log))

        if not message.startswith("SERVER_CONFIRM_RECEIVED_MODEL") or message.split(':::')[1] != epoch_log['local_model_payload_hash']:
            raise Exception("Not receiving the right signal confirmation from server for params transfer")
        else:
            print("Complete transfer local model params")

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
                global_model_params = self.receive_global_model(server_status['global_model_size'])
                self.peripheral_fl.spawn_new_local_training(server_status['global_training_epoch'], global_model_params)
                client_status = {
                    'epoch': self.peripheral_fl.current_training_epoch,
                    'status': 'TRAINING_IN_PROGRESS'
                }
                self.send_to_server(f"CLIENT_UPDATE_STATUS:::{json.dumps(client_status)}")
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
    
    def frequent_status_check(self):
        while True:
            try:
                self.send_to_server(f"CLIENT_INITIATE_STATUS_CHECK")
            except Exception as e:
                print(f"Failed status check {e}")
            time.sleep(30)

    def listen_to_server(self):
        thread = threading.Thread(target=self.frequent_status_check, args=())
        thread.start()
        while not self.break_connection:
            try:
                message = self.sock.recv(1024).decode('utf-8')
                if message:
                    print(f"Received from server: {message}")
                    if message.startswith("SERVER_SEND_STATUS"):
                        self.handle_server_send_status(message)
                    else:
                        self.handle_unrecognized_message(message)

                else:
                    print("Connection closed by the server.")
                    break
            except Exception as e:
                print(f"Error receiving data: {e}")
                break

    def send_to_server(self, message):
        print(f"Sending to server: {message}")
        self.transfer_data_to_server(message.encode('utf-8'))


    def transfer_data_to_server(self, data):
        with self.socket_lock:
            self.sock.sendall(data)

    def close(self):
        self.sock.close()
        print("Connection closed.")

if __name__ == "__main__":
    multiprocessing.set_start_method('forkserver')
    HOST = '127.0.0.1'  # The server's hostname or IP address
    PORT = 65432        # The port used by the server
    
    client = Client(HOST, PORT)
    client.connect()
