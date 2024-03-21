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
import os
import pickle

import cifar10
from fedlearn import sha256_hash, fedavg_aggregate, set_parameters, get_parameters
 
DEVICE = torch.device("cpu" if torch.cuda.is_available() else "cpu")

SERVER_DATABASE_PATH = 'database/server_database.pkl'
CONNECTION_DATABASE_PATH = 'database/connection_database.json'
AGGREGATION_DATABASE_PATH = 'database/aggregation_database.pkl'
SERVER_LOG_DATABASE_PATH = 'database/server_log_database.json'


def write_global_log(global_model_epoch, global_loss, global_accuracy):
    if not os.path.isfile(SERVER_LOG_DATABASE_PATH):
        data = []
        with open(SERVER_LOG_DATABASE_PATH, 'w') as file:
            json.dump([], file)
    else:
        with open(SERVER_LOG_DATABASE_PATH, 'r') as file:
            data = json.load(file)

    data.append({
        'global_model_epoch': global_model_epoch,
        'global_loss': global_loss,
        'global_accuracy': global_accuracy,
    })

    with open(SERVER_LOG_DATABASE_PATH, 'w') as file:
        json.dump(data, file)

def write_global_model(global_model_epoch, global_model_params, global_loss, global_accuracy):
    global_model_params_payload = msgpack.packb(global_model_params, default=msgpack_numpy.encode)
    global_model_size = len(global_model_params_payload)
    with open(AGGREGATION_DATABASE_PATH, 'wb') as file:
        pickle.dump({
            'global_model_epoch': global_model_epoch,
            'global_model_params': global_model_params,
            'global_model_params_payload': global_model_params_payload,
            'global_model_size': global_model_size,
            'global_loss': global_loss,
            'global_accuracy': global_accuracy,
        }, file)

    write_global_log(global_model_epoch, global_loss, global_accuracy)

def read_global_model():
    with open(AGGREGATION_DATABASE_PATH, 'rb') as file:
        data = pickle.load(file)

    return data

def read_or_initialize_global_model():
    if not os.path.isfile(AGGREGATION_DATABASE_PATH):
        write_global_model(
            -1, get_parameters(cifar10.load_model().to(DEVICE)), None, None
        )
    
    return read_global_model()

def evaluate(global_model_params):
    test_loader = cifar10.load_test_data()
    global_model = cifar10.load_model().to(DEVICE)

    set_parameters(global_model, global_model_params)
    loss, accuracy = cifar10.test(global_model, test_loader, DEVICE)

    return loss, accuracy

def centralized_aggregation(current_training_epoch, client_model_record):
    client_results = [client_model_record[uid] for uid in client_model_record.keys()]
    global_model_params = fedavg_aggregate(client_results)
    global_loss, global_accuracy = evaluate(global_model_params)

    print(f"Aggregated result: Training epoch {current_training_epoch} Loss {global_loss}, Acc {global_accuracy}")

    write_global_model(current_training_epoch, global_model_params, global_loss, global_accuracy)

class CentralizeFL():
    def __init__(self) -> None:

        gm_data = read_or_initialize_global_model()
        self.populate_global_model(gm_data)
        self.global_aggregation_process = None
        self.global_model_lock = threading.Lock()

        self.current_training_epoch = self.global_model_epoch + 1
        self.client_model_record = {}
        self.client_result_lock = threading.Lock()

        self.min_fit_clients = 2

    def populate_global_model(self, gm_data=None):
        self.global_model_epoch = gm_data['global_model_epoch']
        self.global_model_params = gm_data['global_model_params']
        self.global_model_params_payload = gm_data['global_model_params_payload']
        self.global_model_size = gm_data['global_model_size']
        self.global_accuracy = gm_data['global_accuracy']
        self.global_loss = gm_data['global_loss']

    def is_aggregation_running(self):
        return (self.global_aggregation_process is not None and self.global_aggregation_process.is_alive())

    def start_aggregation_if_suffient_result(self):
        print("Checking model aggregation condition")
        if len(self.client_model_record.keys()) >= self.min_fit_clients:
            if self.is_aggregation_running():
                raise Exception("A global aggregation process is already running. Something is wrong with the code")
            
            # Delete existing global model
            self.global_model_epoch = None
            self.global_model_params = None 
            self.global_model_params_payload = None
            self.global_model_size = None
            self.global_accuracy = None

            print(f"Starting centralized model aggregation.")
            self.global_aggregation_process = multiprocessing.Process(target=centralized_aggregation, args=(self.current_training_epoch, self.client_model_record,))
            self.global_aggregation_process.start()

    def get_aggregated_model_if_havent(self):
        with self.client_result_lock:
            if self.global_model_params is None:
                gm_data = read_global_model()
                self.populate_global_model(gm_data)
                
                self.current_training_epoch = self.global_model_epoch + 1
                
                print(f"AFTER get_aggregated_model_if_havent: {self.current_training_epoch, self.global_model_epoch, self.global_model_size, self.global_accuracy, self.global_loss,}")

    def receive_client_result(self, client_uid, client_model):
        with self.client_result_lock:
            self.client_model_record[client_uid] = client_model['params']
            self.start_aggregation_if_suffient_result()
        
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

        return "MODEL_RECEIVED"

    def send_status_check(self, client_uid):
        if self.centralized_fl.is_aggregation_running():
            server_status = {
                'client_uid': client_uid,
                'aggregation_running': True
            }
        else:
            self.centralized_fl.get_aggregated_model_if_havent()

            server_status = {
                'client_uid': client_uid,
                'global_training_epoch': self.centralized_fl.current_training_epoch,
                'global_model_size': self.centralized_fl.global_model_size,
                'aggregation_running': False,
                'received_model_from_client': (client_uid in self.centralized_fl.client_model_record)
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