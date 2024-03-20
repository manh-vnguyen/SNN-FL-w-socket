import socket
import threading
import uuid
import msgpack
import msgpack_numpy
import multiprocessing
import time
import copy
import torch
import json
import sys
import hashlib

import cifar10

from typing import Dict, List, Optional, Tuple, Any
from collections import OrderedDict
import numpy as np
import numpy.typing as npt
from functools import reduce

NDArray = npt.NDArray[Any]
NDArrayInt = npt.NDArray[np.int_]
NDArrayFloat = npt.NDArray[np.float_]
NDArrays = List[NDArray]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sha256_hash(data):
    hash_object = hashlib.sha256()
    hash_object.update(data)
    
    return hash_object.hexdigest()

class CentralizeFL():
    def __init__(self) -> None:
        self.test_loader = cifar10.load_test_data()

        self.global_model_params = self.get_parameters(cifar10.load_model().to(DEVICE))
        self.global_model_params_payload = msgpack.packb(self.global_model_params, default=msgpack_numpy.encode)
        self.global_model_size = len(self.global_model_params_payload)
        
        self.global_aggregation_process = None

        self.current_training_epoch = 0
        self.client_model_record = {}
        self.client_model_logs = {}
        self.client_result_lock = threading.Lock()

        self.min_fit_clients = 5

    def get_parameters(self, net) -> List[np.ndarray]:
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, net, parameters: List[np.ndarray]):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v.astype(float)) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def evaluate_global_model(self):
        global_model = cifar10.load_model().to(DEVICE)
        self.set_parameters(global_model, self.global_model_params)
        loss, accuracy = cifar10.test(global_model, self.test_loader, DEVICE)

        return loss, accuracy

    def fedavg_aggregate(self, results: List[Tuple[NDArrays, int]]) -> NDArrays:
        """Compute weighted average."""
        # Calculate the total number of examples used during training
        num_examples_total = sum([num_examples for _, num_examples in results])

        # Create a list of weights, each multiplied by the related number of examples
        weighted_weights = [
            [layer * num_examples for layer in weights] for weights, num_examples in results
        ]

        # Compute average weights of each layer
        weights_prime: NDArrays = [
            reduce(np.add, layer_updates) / num_examples_total
            for layer_updates in zip(*weighted_weights)
        ]
        return weights_prime
    
    def centralized_aggregation(self, data):
        client_model_record = data['client_model_record']
        self.updated_global_model = self.fedavg_aggregate(client_model_record)
        self.evaluate(self.updated_global_model)
    
    def is_aggregation_running(self):
        return (self.global_aggregation_process is not None and self.global_aggregation_process.is_alive())

    def start_aggregation_if_suffient_result(self):
        print("Checking model aggregation condition")
        if len(self.client_model_record.keys()) >= self.min_fit_clients:
            if self.is_aggregation_running():
                raise Exception("A global aggregation process is already running. Something is wrong with the code")
            
            data = {
                "current_training_epoch", self.current_training_epoch,
                "client_model_record", self.client_model_record,
                "client_model_logs", self.client_model_logs,
            }

            print(f"Starting centralized model aggregation.")
            self.global_aggregation_process = multiprocessing.Process(target=self.centralized_aggregation, args=(data))
            self.global_aggregation_process.start()

            self.current_training_epoch += 1
            self.client_model_record = {}
            self.client_model_logs = {}

    def receive_client_result(self, client_model_params):
        print(f"Let's do nothing rightnow. {len(client_model_params)}")
        # with self.client_result_lock:
        #     self.client_model_record[client_uid] = client_model_params
        #     self.client_model_logs[client_uid] = client_model_log

        #     self.start_aggregation_if_suffient_result()
    

class Server:
    def __init__(self, host='127.0.0.1', port=65432):
        self.host = host
        self.port = port
        self.clients = {}  # Maps UIDs to client sockets
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))

        self.client_socket_lock = threading.Lock()

        self.client_status = {
            'latest_client_result_epoch': None
        }

        self.centralized_fl = CentralizeFL()
        

    def identify_client(self, client_socket):
        client_socket.send(f"SERVER_IDENTIFY_CLIENT".encode('utf-8'))

    def handle_unrecognized_message(self, message):
        print(f"Unable to recognize message: {message}")

    def receive_data_from_client(self, data_size):
        print("Trying to recieve model data from client")
        received_data = b''
        while len(received_data) < data_size:
            try:
                more_data = self.client_socket.recv(data_size - len(received_data))
                if not more_data:
                    raise Exception("Server closed the connection unexpectedly.")
                received_data += more_data
            except socket.timeout:
                raise Exception("Timed out waiting for data from server.")
            
        return received_data

    def receive_client_training_result(self, message):
        client_model_size = int(message.split(':::')[1])
        self.send_to_client(f"SERVER_READY_TO_RECEIVE:::{client_model_size}")

        with open("debug_log", "w") as file:
            file.write(f"Receiving client model of size: {client_model_size}")

        client_model_payload = self.receive_data_from_client(client_model_size)
        client_model = msgpack.unpackb(client_model_payload, object_hook=msgpack_numpy.decode)

        self.send_to_client(f"SERVER_CONFIRM_RECEIVED_MODEL:::{sha256_hash(client_model_payload)}")

        client_epoch = client_model['epoch']
        client_model_params = client_model['params']
        
        self.centralized_fl.receive_client_result(client_model_params)

        return "MODEL_RECEIVED"

    def send_status_check(self):
        server_status = {
            'global_training_epoch': self.centralized_fl.current_training_epoch,
            'global_model_size': self.centralized_fl.global_model_size,
            'aggregation_running': self.centralized_fl.is_aggregation_running(),
            'latest_client_result_epoch': self.client_status['latest_client_result_epoch']
        }

        self.send_to_client(f"SERVER_SEND_STATUS:::{json.dumps(server_status)}")

    def send_global_model_to_client(self, message):
        if int(message.split(':::')[1]) != self.centralized_fl.global_model_size:
            raise Exception("Something wrong with confirming the global model size. Can't transfer model")

        self.transfer_data_to_client(self.centralized_fl.global_model_params_payload)

        message = self.client_socket.recv(1024).decode('utf-8')

        if not message.startswith("CLIENT_CONFIRM_MODEL_RECEIVED") or message.split(":::")[1] != sha256_hash(self.centralized_fl.global_model_params_payload):
            raise Exception("The client-received model does not match")
        else:
            print("Done sending model to client")

    def handle_client_connection(self, client_socket, client_addr):
        # client_socket.settimeout(5.0)
        client_uid = None

        while True:
            try:
                message = client_socket.recv(1024).decode('utf-8')
                if message:
                    print(f"Message from {client_addr}: {message}")
                    
                    if message == "CLIENT_INITIATE_STATUS_CHECK":
                        self.send_status_check()
                    elif message.startswith("CLIENT_INITIATE_LOCAL_PARAMS_TRANSFER"):
                        self.receive_client_training_result(message)
                    elif message.startswith("CLIENT_INITIATE_GLOBAL_MODEL_RECEIVE"):
                        self.send_global_model_to_client(message)
                    else:
                        self.handle_unrecognized_message(message)
                else:
                    break
            except ConnectionResetError:
                break

        print(f"Connection closed. Address: {client_addr}. Client: {client_uid}")
        client_socket.close()

    def send_to_client(self, message):
        self.transfer_data_to_client(message.encode('utf-8'))

    def transfer_data_to_client(self, data):
        with self.client_socket_lock:
            self.client_socket.sendall(data)
        
    def run(self):
        self.server_socket.listen()
        print(f"Server listening on {self.host}:{self.port}")
        print("Server running. Ctrl+C to stop.")

        try:
            while True:
                self.client_socket, self.client_addr = self.server_socket.accept()

                self.handle_client_connection(self.client_socket, self.client_addr)

        except KeyboardInterrupt:
            print("Server stopping...")
        finally:
            self.server_socket.close()

if __name__ == "__main__":
    server = Server()
    server.run()
