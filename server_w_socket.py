import socket
import threading
import uuid
import msgpack
import msgpack_numpy
from torch import multiprocessing
import time
import copy
import torch
import json
import sys
import hashlib
import os
import pickle

import cifar10

from typing import Dict, List, Optional, Tuple, Any
from collections import OrderedDict
import numpy as np
import numpy.typing as npt
from functools import reduce
from io import BytesIO
from dataclasses import dataclass
from typing import cast

NDArray = npt.NDArray[Any]
NDArrayInt = npt.NDArray[np.int_]
NDArrayFloat = npt.NDArray[np.float_]
NDArrays = List[NDArray]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SERVER_DATABASE_PATH = 'database/server_database.pkl'
CONNECTION_DATABASE_PATH = 'database/connection_database.json'

def sha256_hash(data):
    hash_object = hashlib.sha256()
    hash_object.update(data)
    
    return hash_object.hexdigest()

@dataclass
class Parameters:
    """Model parameters."""

    tensors: List[bytes]
    tensor_type: str

def ndarray_to_bytes(ndarray: NDArray) -> bytes:
    """Serialize NumPy ndarray to bytes."""
    bytes_io = BytesIO()
    # WARNING: NEVER set allow_pickle to true.
    # Reason: loading pickled data can execute arbitrary code
    # Source: https://numpy.org/doc/stable/reference/generated/numpy.save.html
    np.save(bytes_io, ndarray, allow_pickle=False)
    return bytes_io.getvalue()

def bytes_to_ndarray(tensor: bytes) -> NDArray:
    """Deserialize NumPy ndarray from bytes."""
    bytes_io = BytesIO(tensor)
    # WARNING: NEVER set allow_pickle to true.
    # Reason: loading pickled data can execute arbitrary code
    # Source: https://numpy.org/doc/stable/reference/generated/numpy.load.html
    ndarray_deserialized = np.load(bytes_io, allow_pickle=False)
    return cast(NDArray, ndarray_deserialized)

def ndarrays_to_parameters(ndarrays: NDArrays) -> Parameters:
    """Convert NumPy ndarrays to parameters object."""
    tensors = [ndarray_to_bytes(ndarray) for ndarray in ndarrays]
    return Parameters(tensors=tensors, tensor_type="numpy.ndarray")

def parameters_to_ndarrays(parameters: Parameters) -> NDArrays:
    """Convert parameters object to NumPy ndarrays."""
    return [bytes_to_ndarray(tensor) for tensor in parameters.tensors]

def get_parameters(net) -> List[np.ndarray]:
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v.astype(float)) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

def evaluate(global_model_params):
    test_loader = cifar10.load_test_data()
    global_model = cifar10.load_model().to(DEVICE)

    set_parameters(global_model, global_model_params)
    loss, accuracy = cifar10.test(global_model, test_loader, DEVICE)

    return loss, accuracy

def fedavg_aggregate(results: List[Tuple[NDArrays, int]]) -> NDArrays:
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

def centralized_aggregation(client_model_record):
    client_results = [client_model_record[uid] for uid in client_model_record.keys()]
    agged_model_params = parameters_to_ndarrays(ndarrays_to_parameters(fedavg_aggregate(client_results)))
    global_loss, global_acc = evaluate(agged_model_params)
    print(f"Aggregated result: Loss {global_loss}, Acc {global_acc}")

class CentralizeFL():
    def __init__(self) -> None:

        database = self.read_or_initiate_database()

        current_training_epoch = database['current_training_epoch']
        global_model_params = database['global_model_params']
        client_model_record = database['client_model_record']
        client_model_logs = database['client_model_logs']

        self.global_model_params = self.get_parameters(cifar10.load_model().to(DEVICE)) if global_model_params is None else global_model_params
        self.global_model_params_payload = msgpack.packb(self.global_model_params, default=msgpack_numpy.encode)
        self.global_model_size = len(self.global_model_params_payload)
        
        self.global_aggregation_process = None

        self.current_training_epoch = 0 if current_training_epoch is None else current_training_epoch
        self.client_model_record = {} if client_model_record is None else client_model_record
        self.client_model_logs = {} if client_model_logs is None else client_model_logs
        self.client_result_lock = threading.Lock()

        self.min_fit_clients = 5

        print(f"number of client results: {len(self.client_model_record)}")

    def write_database(self, data):
        with open(SERVER_DATABASE_PATH, 'wb') as file:
            pickle.dump(data, file)

    def read_or_initiate_database(self):
        if not os.path.isfile(SERVER_DATABASE_PATH):
            self.write_database({
                'current_training_epoch': 0,
                'global_model_params': None,
                'client_model_record': None, 
                'client_model_logs': None,
            })

        with open(SERVER_DATABASE_PATH, 'rb') as file:
            data = pickle.load(file)
        
        return data
    
    def make_backup(self):
        self.write_database({
            'current_training_epoch': self.current_training_epoch,
            'global_model_params': self.global_model_params,
            'client_model_record': self.client_model_record, 
            'client_model_logs': self.client_model_logs,
        })
    
    def is_aggregation_running(self):
        return (self.global_aggregation_process is not None and self.global_aggregation_process.is_alive())

    def start_aggregation_if_suffient_result(self):
        print("Checking model aggregation condition")
        if len(self.client_model_record.keys()) >= self.min_fit_clients:
            if self.is_aggregation_running():
                raise Exception("A global aggregation process is already running. Something is wrong with the code")

            print(f"Starting centralized model aggregation.")
            self.global_aggregation_process = multiprocessing.Process(target=centralized_aggregation, args=(copy.deepcopy(self.client_model_record),))
            self.global_aggregation_process.start()

            # self.current_training_epoch += 1
            # self.client_model_record = {}
            

    def receive_client_result(self, client_uid, client_model):
        with self.client_result_lock:
            self.client_model_record[client_uid] = client_model['params']
            if client_uid not in self.client_model_logs:
                self.client_model_logs[client_uid] = []
            self.client_model_logs[client_uid].append({
                'epoch': client_model['epoch'],
                'accuracy': client_model['accuracy'],
                'loss': client_model['loss'],
            })
        
            self.start_aggregation_if_suffient_result()
        
        self.make_backup()
    

class Server:
    def __init__(self, host='127.0.0.1', port=65432):
        self.host = host
        self.port = port

        connection_database = self.read_or_initiate_database()

        self.clients = {}  # Maps UIDs to client sockets
        for uid in connection_database:
            self.clients[uid] = {
                'socket': None,
                'connection_status': 'DISCONNECTED'
            }

        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))

        self.centralized_fl = CentralizeFL()

    def write_database(self, data):
        with open(CONNECTION_DATABASE_PATH, 'w') as file:
            json.dump(data, file)

    def read_or_initiate_database(self):
        if not os.path.isfile(CONNECTION_DATABASE_PATH):
            data = []
            self.write_database(data)
        else:
            with open(CONNECTION_DATABASE_PATH, 'r') as file:
                data = json.load(file)
        
        return data
    
    def make_backup(self):
        self.write_database(list(self.clients.keys()))

    def handle_unrecognized_message(self, message):
        print(f"Unable to recognize message: {message}")

    def receive_data_from_specific_client(self, client_uid, data_size):
        received_data = b''
        while len(received_data) < data_size:
            try:
                more_data = self.clients[client_uid]['socket'].recv(data_size - len(received_data))
                if not more_data:
                    raise Exception("Server closed the connection unexpectedly.")
                received_data += more_data
            except socket.timeout:
                raise Exception("Timed out waiting for data from server.")
            
        return received_data

    def receive_client_training_result(self, client_uid, message):

        client_model_size = int(message.split(':::')[1])
        self.send_to_specific_client(client_uid, f"SERVER_READY_TO_RECEIVE:::{client_model_size}")

        client_model_payload = self.receive_data_from_specific_client(client_uid, client_model_size)
        client_model = msgpack.unpackb(client_model_payload, object_hook=msgpack_numpy.decode)

        self.send_to_specific_client(client_uid, f"SERVER_CONFIRM_RECEIVED_MODEL:::{sha256_hash(client_model_payload)}")

        if client_model['epoch'] != self.centralized_fl.current_training_epoch:
            print("Mismatch epoch, reject model")
            return "MODEL_REJECTED_MISMATCH_EPOCH"
        if client_uid in self.centralized_fl.client_model_record.keys():
            print("Already exist, reject model")
            return "MODEL_REJECTED_ALREADY_EXIST"
        
        self.centralized_fl.receive_client_result(client_uid, client_model)
        self.make_backup()

        return "MODEL_RECEIVED"

    def send_status_check(self, client_uid):
        server_status = {
            'client_uid': client_uid,
            'global_training_epoch': self.centralized_fl.current_training_epoch,
            'global_model_size': self.centralized_fl.global_model_size,
            'aggregation_running': self.centralized_fl.is_aggregation_running(),
            'received_model_from_client': (client_uid in self.centralized_fl.client_model_logs)
        }

        self.send_to_specific_client(client_uid, f"SERVER_SEND_STATUS:::{json.dumps(server_status)}")

    def send_global_model_to_client(self, client_uid, message):
        if int(message.split(':::')[1]) != self.centralized_fl.global_model_size:
            raise Exception("Something wrong with confirming the global model size. Can't transfer model")

        print("Start transfering")
        start_time = time.time()
        self.transfer_data_to_specific_client(client_uid, self.centralized_fl.global_model_params_payload)
        print(f"Complete global model transfer at: {time.time() - start_time}")

        message = self.clients[client_uid]['socket'].recv(1024).decode('utf-8')

        if not message.startswith("CLIENT_CONFIRM_MODEL_RECEIVED") or message.split(":::")[1] != sha256_hash(self.centralized_fl.global_model_params_payload):
            raise Exception("The client-received model does not match")
        print(f"Complete global model confirm at: {time.time() - start_time}")

    def handle_client_update_status(self, client_uid, message):
        print(f"Receive message from {client_uid}: {message}")
        pass

    def create_new_client(self, client_socket):
        client_uid = str(uuid.uuid4())

        self.clients[client_uid] = {
            'socket': client_socket,
            'connection_status': 'CONNECTED'
        }
        self.make_backup()
        
        client_socket.send(f"SERVER_ASSIGN_NEW_CLIENT_ID:{client_uid}".encode('utf-8'))

        return client_uid

    def update_existing_client(self, client_socket, message):
        client_uid = message.split(':')[1]

        if client_uid in self.clients:
            self.clients[client_uid]['socket'] = client_socket
            self.clients[client_uid]['connection_status'] = 'CONNECTED'

            client_socket.send(f"SERVER_ACK_EXISTING_CLIENT:{client_uid}".encode('utf-8'))

            return client_uid
        else:
            raise Exception(f"Client uid not accepted: {message}")
        
    def check_client_uid_ack(self, client_uid, message):
        ack_uid = message.split(':')[1]
        if ack_uid != client_uid:
            raise Exception(f"Ack client uid failed: {message}")

    def hand_shake(self, client_socket, client_addr):
        # Stage 1: Server ask for identity
        client_socket.send(f"SERVER_IDENTIFY_CLIENT".encode('utf-8'))


        # Stage 2: Client response with existing id or new id request
        message = client_socket.recv(1024).decode('utf-8')
        print(message)
        
        if message == "NEW_CLIENT_REQUEST_ID":
            client_uid = self.create_new_client(client_socket)
        elif message.startswith("EXISTING_CLIENT_SUBMIT_ID"):
            client_uid = self.update_existing_client(client_socket, message)

        # Stage 3: Client response with acknowledging the id
        message = client_socket.recv(1024).decode('utf-8')
        print(message)
        if message.startswith("NEW_CLIENT_SUBMIT_ACK"):
            self.check_client_uid_ack(client_uid, message)
            print(f"New client joined the network. Connection established. UID: {client_uid}")
        elif message.startswith("EXISTING_CLIENT_SUBMIT_ACK"):
            self.check_client_uid_ack(client_uid, message)
            print(f"Existing client reconnected. UID: {client_uid}")
        else:
            raise Exception(f"Handshake failed at: {message}")
        
        return client_uid

    def handle_client_connection(self, client_socket, client_addr):
        # client_socket.settimeout(5.0)
        client_uid = None

        try:
            client_uid = self.hand_shake(client_socket, client_addr)
        except Exception as e:
            print(f"Error: {e}")

        if client_uid is not None:
            while True:
                try:
                    message = client_socket.recv(1024).decode('utf-8')
                    if message:
                        if message == "CLIENT_INITIATE_STATUS_CHECK":
                            self.send_status_check(client_uid)
                        elif message.startswith("CLIENT_UPDATE_STATUS"):
                            self.handle_client_update_status(client_uid, message)
                        elif message.startswith("CLIENT_INITIATE_LOCAL_PARAMS_TRANSFER"):
                            self.receive_client_training_result(client_uid, message)
                        elif message.startswith("CLIENT_INITIATE_GLOBAL_MODEL_RECEIVE"):
                            self.send_global_model_to_client(client_uid, message)
                        else:
                            self.handle_unrecognized_message(message)
                    else:
                        break
                except Exception as e:
                    print(f"Error: {e}")
                    break

        print(f"Connection closed. Address: {client_addr}. Client: {client_uid}")
        if client_uid is not None and client_uid in self.clients:
            del self.clients[client_uid]['socket']
            self.clients[client_uid]['connection_status'] = 'DISCONNECTED'
        client_socket.close()

    def send_to_specific_client(self, uid, message):
        print(f"Sent to {uid}: {message}")
        self.transfer_data_to_specific_client(uid, message.encode('utf-8'))

    def transfer_data_to_specific_client(self, uid, data):
        if uid in self.clients and self.clients[uid]['connection_status'] == 'CONNECTED':
            self.clients[uid]['socket'].sendall(data)
        else:
            print(f"Client UID {uid} not found or DISCONNECTED.")

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

if __name__ == "__main__":
    multiprocessing.set_start_method('forkserver')
    server = Server()
    server.run()
    # centralized_fl = CentralizeFL()
    # centralized_fl.start_aggregation_if_suffient_result()