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

from collections import OrderedDict
from typing import Dict, List, Tuple
import numpy as np
import cifar10

CLIENT_DATABASE_PATH = 'client_database.json'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PeripheralFL():
    def __init__(self) -> None:
        self.node_id = 0
        # self.trainloader, self.testloader = cifar10.load_client_data(self.node_id)

        self.local_model_parameters = None
        self.local_model_params_payload = None
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
        
        self.parent_conn.recv()
        
        # If there are training result, process this result
        with open('/tmp/data/training_result.pkl', 'rb') as file:
            self.local_model_params, loss, accuracy = pickle.load(file)

        self.local_model_params_payload = msgpack.packb(self.local_model_params, default=msgpack_numpy.encode)
        
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



if __name__ == "__main__":
    multiprocessing.set_start_method('forkserver')

    periferal_fl = PeripheralFL()

    parameters = periferal_fl.get_parameters(cifar10.load_model().to(DEVICE), {})
    periferal_fl.spawn_new_local_training(0, parameters)

    periferal_fl.local_traing_process.join()
    print("Training process completed")

    print(periferal_fl.handle_epoch_completed())
