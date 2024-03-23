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
import argparse

from collections import OrderedDict
from typing import Dict, List, Tuple
import numpy as np
import cifar10
from fedlearn import sha256_hash, set_parameters, get_parameters


parser = argparse.ArgumentParser(description="Flower")
parser.add_argument("--node-id", type=int, required=True, choices=range(0, 10))
parser.add_argument("--host", type=str, default="127.0.0.1")
parser.add_argument("--port", type=int, default=65432, choices=range(0, 65536))
ARGS = parser.parse_args()

CLIENT_DATABASE_PATH = f"database/client_database_{ARGS.node_id}.json"
RESULT_CACHE_PATH = f"/tmp/data/training_result_{ARGS.node_id}.pkl"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PeripheralFL():
    def __init__(self, client_logs = []) -> None:
        self.local_model_result = None
        self.local_model_payload = None
        self.parent_conn, self.child_conn = None, None
        
        self.local_traing_process = None

        self.current_training_epoch = None if len(client_logs) == 0 else client_logs[-1]['epoch']
        self.client_logs = client_logs


    def fit(
        self, conn, 
        parameters: List[np.ndarray], 
        config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict]:
        try:
            # Load data
            trainloader, testloader = cifar10.load_client_data(ARGS.node_id)

            # Set model parameters, train model, return updated model parameters
            model = cifar10.load_model().to(DEVICE)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0)

            set_parameters(model, parameters)

            cifar10.train(model, optimizer, trainloader, DEVICE, 1)
            loss, accuracy = cifar10.test(model, testloader, DEVICE)
            trained_local_result = (get_parameters(model), len(trainloader.dataset))

            with open(RESULT_CACHE_PATH, 'wb') as file:
                pickle.dump((trained_local_result, loss, accuracy), file)
            conn.send("SUCCESS")
        except Exception as e:
            conn.send(f"LOCAL TRAINING FAILED {e}")
        finally:
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

        self.local_model_result = None
        self.local_model_payload = None
        self.parent_conn, self.child_conn = None, None
        
        self.local_traing_process = None

    
    def spawn_new_local_training(self, epoch, parameters):
        self.discard_local_training()
        
        self.parent_conn, self.child_conn = multiprocessing.Pipe()
        self.local_traing_process = multiprocessing.Process(target=self.fit, args=(self.child_conn, parameters, {}))

        self.current_training_epoch = epoch
        self.local_traing_process.start()

    def did_local_training_suceed(self):
        if len(self.client_logs) == 0 or self.client_logs[-1]['epoch'] != self.current_training_epoch:
            if self.parent_conn is None or not self.parent_conn.poll():
                return False
            else:
                training_message = self.parent_conn.recv()
                if training_message != 'SUCCESS':
                    return False
                
        return True
    
    def load_local_training_result(self):
        with open(RESULT_CACHE_PATH, 'rb') as file:
            self.local_model_result, self.local_model_loss, self.local_model_accuracy = pickle.load(file)

        self.local_model_payload = msgpack.packb({
                'epoch': self.current_training_epoch,
                'params': self.local_model_result, 
                'accuracy': self.local_model_accuracy,
                'loss': self.local_model_loss,
            }, default=msgpack_numpy.encode)

    def get_local_training_result_log(self):
        if self.local_model_result is None:
            self.load_local_training_result()

        if len(self.client_logs) > 0 and self.client_logs[-1]['epoch'] == self.current_training_epoch:
            return self.client_logs[-1]
        
        self.client_logs.append({
            'epoch': self.current_training_epoch,
            'status': 'TRAINING_COMPLETED',
            'loss': self.local_model_loss,
            'accuracy': self.local_model_accuracy, 
            'local_model_payload_size': len(self.local_model_payload),
            'local_model_payload_hash': sha256_hash(self.local_model_payload)
        })
        return self.client_logs[-1]

class Client:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self.database = self.read_or_initiate_database()
        self.client_uid = self.database['client_uid']
        self.hand_shake_completed = False

        self.peripheral_fl = PeripheralFL(self.database['client_logs'])
        # self.sock.settimeout(5.0)

    def write_database(self, data):
        with open(CLIENT_DATABASE_PATH, 'w') as file:
            json.dump(data, file)

    def read_or_initiate_database(self):
        if not os.path.isfile(CLIENT_DATABASE_PATH):
            self.write_database({
                'client_uid': None,
                'client_logs': []
            })

        with open(CLIENT_DATABASE_PATH, 'r') as file:
            data = json.load(file)
        
        return data
    
    def make_backup(self):
        self.write_database({
            'client_uid': self.client_uid,
            'client_logs': self.peripheral_fl.client_logs
        })

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

        return True

    def handle_new_identity_assigned(self, message):
        if self.client_uid is not None:
            print(f"Server attempts to assign new id on existing client: {self.client_uid}")
            return False
        else:
            self.client_uid = message.split(':')[1]
            print(f"Server assigned client_uid: {self.client_uid}. Connection established.")
            self.send_to_server(f"NEW_CLIENT_SUBMIT_ACK:{self.client_uid}")
            return True

    def handle_existing_client_ack(self, message):
        if self.client_uid != message.split(':')[1]:
            print(f"Server acknowledged wrong client id")
            return False
        else:
            print(f"Connection with server established")
            self.send_to_server(f"EXISTING_CLIENT_SUBMIT_ACK:{self.client_uid}")
            return True

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

        # A separate listen from the listen_to_server while loop
        message = self.sock.recv(1024).decode('utf-8')
        print(f"Receiving from server: {message}")

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

    def start_new_local_training(self, server_status):
        global_model_params = self.receive_global_model(server_status['global_model_size'])
        self.peripheral_fl.spawn_new_local_training(server_status['global_training_epoch'], global_model_params)
        client_status = {
            'epoch': self.peripheral_fl.current_training_epoch,
            'status': 'TRAINING_IN_PROGRESS'
        }
        self.send_to_server(f"CLIENT_UPDATE_STATUS:::{json.dumps(client_status)}")
        self.make_backup()

    def is_local_training_behind(self, server_status):
        return server_status['global_training_epoch'] != self.peripheral_fl.current_training_epoch

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
            if self.is_local_training_behind(server_status):
                self.start_new_local_training(server_status)
            else:
                if self.peripheral_fl.is_running_local_training():
                    client_status = {
                        'epoch': self.peripheral_fl.current_training_epoch,
                        'status': 'TRAINING_IN_PROGRESS'
                    }
                    self.send_to_server(f"CLIENT_UPDATE_STATUS:::{json.dumps(client_status)}")
                else:
                    if not server_status['received_model_from_client']:
                        if self.peripheral_fl.did_local_training_suceed():
                            epoch_log = self.peripheral_fl.get_local_training_result_log()
                            self.make_backup()
                            self.local_params_transfer(epoch_log)
                        else:
                            self.start_new_local_training(server_status)
                    else:
                        client_status = {
                            'epoch': self.peripheral_fl.current_training_epoch,
                            'status': 'AWAIT_NEW_GLOBAL_MODEL'
                        }
                        self.send_to_server(f"CLIENT_UPDATE_STATUS:::{json.dumps(client_status)}")
    
    def hand_shake(self):
        message = self.sock.recv(1024).decode('utf-8')
        print(f"Received from server: {message}")
        if not message == "SERVER_IDENTIFY_CLIENT" or not self.handle_identity_prompt():
            raise Exception(f"Hand shake failed at: {message}")
        
        message = self.sock.recv(1024).decode('utf-8')
        if message.startswith("SERVER_ASSIGN_NEW_CLIENT_ID"):
            if not self.handle_new_identity_assigned(message):
                raise Exception(f"Hand shake failed at: {message}")
        elif message.startswith("SERVER_ACK_EXISTING_CLIENT"):
            if not self.handle_existing_client_ack(message):
                raise Exception(f"Hand shake failed at: {message}")
        
        self.hand_shake_completed = True
        
        # self.handle_unrecognized_message(message)

    def frequent_status_check(self):
        self.send_to_server(f"CLIENT_INITIATE_STATUS_CHECK")

        message = self.sock.recv(1024).decode('utf-8')
        if not message.startswith("SERVER_SEND_STATUS"):
            raise Exception("Client status check not receiving correct response")
        
        self.handle_server_send_status(message)

    def listen_to_server(self):
        while True:
            if not self.hand_shake_completed:
                self.hand_shake()
                self.make_backup()
            else:
                self.frequent_status_check()
            time.sleep(5)

    def send_to_server(self, message):
        print(f"Sending to server: {message}")
        self.transfer_data_to_server(message.encode('utf-8'))


    def transfer_data_to_server(self, data):
        self.sock.sendall(data)

    def close(self):
        self.sock.close()
        print("Connection closed.")

if __name__ == "__main__":
    multiprocessing.set_start_method('forkserver')
    
    client = Client(ARGS.host, ARGS.port)
    client.connect()



    # peripheral_fl = PeripheralFL()

    # params = get_parameters(cifar10.load_model().to(DEVICE))
    # peripheral_fl.spawn_new_local_training(0, params)