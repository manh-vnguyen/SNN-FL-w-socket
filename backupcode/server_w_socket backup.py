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

class CentralizeFL():
    def __init__(self) -> None:
        self.test_loader = cifar10.load_test_data()

        self.global_model_params = self.get_parameters(cifar10.load_model().to(DEVICE))
        self.global_model_params_payload = msgpack.packb(self.centralized_fl.global_model_params, default=msgpack_numpy.encode)
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

    def start_aggregation_process(self):
        if self.is_aggregation_running():
            raise Exception("A global aggregation process is already running.")
        
        data = {
            "current_training_epoch", self.current_training_epoch,
            "client_model_record", self.client_model_record,
            "client_model_logs", self.client_model_logs,
        }

        print(f"Starting centralized model aggregation.")
        self.global_aggregation_process = multiprocessing.Process(target=self.centralized_aggregation, args=(data))
        self.global_aggregation_process.start()

        with self.client_result_lock:
            self.current_training_epoch += 1
            self.client_model_record = {}
            self.client_model_logs = {}

    def check_aggregation_condition(self):
        print("Checking model aggregation condition")
        if len(self.client_model_record.keys()) >= self.min_fit_clients:
            self.start_aggregation_process()

    def receive_client_result(self, client_uid, client_model_params, client_model_log):
        with self.client_result_lock:
            self.client_model_record[client_uid] = client_model_params
            self.client_model_logs[client_uid] = client_model_log

        self.check_aggregation_condition()
    

class Server:
    def __init__(self, host='127.0.0.1', port=65432):
        self.host = host
        self.port = port
        self.clients = {}  # Maps UIDs to client sockets
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))

        self.centralized_fl = CentralizeFL()
        

    def identify_client(self, client_socket):
        client_socket.send(f"SERVER_IDENTIFY_CLIENT".encode('utf-8'))

    def create_new_client(self, client_socket):
        client_uid = str(uuid.uuid4())

        self.clients[client_uid] = {
            'socket': client_socket,
            'lock': threading.Lock(),
            'connection_status': 'CONNECTED',
            'latest_client_result_epoch': None
        }
        
        client_socket.send(f"SERVER_ASSIGN_NEW_CLIENT_ID:{client_uid}".encode('utf-8'))

        return client_uid

    def update_existing_client(self, client_socket, message):
        client_uid = message.split(':')[1]

        if client_uid in self.clients:
            self.clients[client_uid]['socket'] = client_socket
            self.clients[client_uid]['lock'] = threading.Lock()
            self.clients[client_uid]['connection_status'] = 'CONNECTED'

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

    def receive_data_from_specific_client(self, client_uid, data_size):
        client_socket = self.clients[client_uid]

        received_data = b''
        while len(received_data) < data_size:
            try:
                more_data = client_socket.recv(data_size - len(received_data))
                if not more_data:
                    raise Exception("Server closed the connection unexpectedly.")
                received_data += more_data
            except socket.timeout:
                raise Exception("Timed out waiting for data from server.")
            
        return received_data
    
    def record_client_model(self, client_uid, client_model):
        client_epoch = client_model['epoch']
        client_model_params = client_model['params']
        client_model_log = client_model['log']
        if client_epoch != self.centralized_fl.current_training_epoch:
            return "MODEL_REJECTED_MISMATCH_EPOCH"
        if client_uid in self.centralized_fl.client_model_record.keys():
            return "MODEL_REJECTED_ALREADY_EXIST"
        
        self.centralized_fl.receive_client_result(client_uid, client_model_params, client_model_log)

        return "MODEL_RECEIVED"


    def handle_client_model(self, message, client_uid):

        client_model_size = int(message.split(':')[1])
        self.send_to_specific_client(self, client_uid, "SERVER_READY_TO_RECEIVE")

        client_model = msgpack.unpackb(self.receive_data_from_specific_client(client_uid, client_model_size))

        self.record_client_model(client_model)

    def handle_client_connection(self, client_socket, client_addr):
        # client_socket.settimeout(5.0)
        client_uid = None
        self.identify_client(client_socket)

        while True:
            try:
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
                        self.handle_client_model(message, client_uid)

                    else:
                        self.handle_unrecognized_message(client_socket, message)
                else:
                    break
            except ConnectionResetError:
                break

        print(f"Connection closed. Address: {client_addr}. Client: {client_uid}")
        if client_uid in self.clients:
            del self.clients[client_uid]['socket']
            del self.clients[client_uid]['lock']
            self.clients[client_uid]['connection_status'] = 'DISCONNECTED'
        client_socket.close()

    def send_to_all_clients(self, message):
        for uid in self.clients.keys():
            self.send_to_specific_client(uid, message)

    def send_to_specific_client(self, uid, message):
        if uid in self.clients and self.clients[uid]['connection_status'] == 'CONNECTED':
            with self.clients[uid]['lock']:
                self.clients[uid]['socket'].send(message.encode('utf-8'))
            print(f"Sent to {uid}: {message}")
        else:
            print(f"Client UID {uid} not found or DISCONNECTED.")


    def frequent_status_check(self):
        while True:
            try:
                for uid in self.clients.keys():
                    if self.clients['connection_status'] == 'DISCONNECTED':
                        continue
                    status = {
                        'global_training_epoch': self.centralized_fl.current_training_epoch,
                        'global_model_size': self.centralized_fl.global_model_size,
                        'aggregation_running': self.centralized_fl.is_aggregation_running(),
                        'latest_client_result_epoch': self.clients[uid]['latest_client_result_epoch']
                    }

                    self.send_to_specific_client(uid, f"SERVER_SEND_STATUS:::{json.dumps(status)}")
            except Exception as e:
                print(f"Failed status check {e}")
            time.sleep(30)


    def run(self):
        self.server_socket.listen()
        print(f"Server listening on {self.host}:{self.port}")
        print("Server running. Ctrl+C to stop.")

        thread = threading.Thread(target=self.frequent_status_check, args=())
        thread.start()
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
    server = Server()
    server.run()

    # server_thread = threading.Thread(target=server.run)
    # server_thread.start()

    # try:
    #     while True:
    #         uid = input("Enter the UID of the client to send a message: ")
    #         if uid.lower() == 'quit':
    #             break
    #         message = input("Enter a message to send to the client: ")
    #         server.send_to_specific_client(uid, message)
    # except KeyboardInterrupt:
    #     pass
    # finally:
    #     print("Server shutdown.")
