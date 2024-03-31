
from torch import multiprocessing
import time
import torch
import json
import os
import pickle
import argparse
from dotenv import load_dotenv
from typing import Dict, List, Tuple
import numpy as np
import random
import uuid

parser = argparse.ArgumentParser(description="Flower")
parser.add_argument("--db_postfix", type=str, default="")
ARGS = parser.parse_args()

load_dotenv(f"database{ARGS.db_postfix}/.env")

if os.getenv('MODEL') == 'SNN':
    import SNN as NN
elif os.getenv('MODEL') == 'ANN':
    import ANN as NN
else:
    raise Exception("Model type wrong!!")

NOISE_ABS_STD =  None if os.getenv('NOISE').split(',')[0] == '_' else float(os.getenv('NOISE').split(',')[0])
NOISE_PERCENTAGE_STD = None if os.getenv('NOISE').split(',')[1] == '_' else float(os.getenv('NOISE').split(',')[1])
PARAMS_COMPRESSION_RATE = None if os.getenv('PARAMS_COMPRESSION_RATE') == '_' else float(os.getenv('PARAMS_COMPRESSION_RATE'))

from fedlearn import set_parameters, get_parameters, add_percentage_gaussian_noise_to_model, add_constant_gaussian_noise_to_model, fedavg_aggregate, compress_parameters

gpu_assignment = [int(x) for x in os.getenv('GPU_ASSIGNMENT').split(',')]

SERVER_LOG_DATABASE_PATH = f"database{ARGS.db_postfix}/server_log_database.json"

TEMP_GLOBAL_MODEL_PATH = f"database{ARGS.db_postfix}/global_model_temp.pkl"
PERMANENT_GLOBAL_MODEL_PATH = f"database{ARGS.db_postfix}/global_model_permanent.pkl"

DEVICE = torch.device(f"cuda:{os.getenv('SERVER_GPU_ASSIGNMENT')}" if torch.cuda.is_available() else "cpu")
CPU_DEVICE = torch.device("cpu")

class Client:
    def __init__(self, node_id) -> None:
        self.node_id = node_id
        self.device = torch.device(f"cuda:{gpu_assignment[self.node_id]}" if torch.cuda.is_available() else "cpu")
        self.result_cach_path = f"database{ARGS.db_postfix}/training_result_{self.node_id}.pkl"
        self.client_database_path = f"database{ARGS.db_postfix}/client_database_{self.node_id}.json"
        self.database = self.read_or_initiate_database()
        self.client_uid = self.database['client_uid']
        self.client_logs = self.database['client_logs']
        # self.sock.settimeout(5.0)

    def write_database(self, data):
        with open(self.client_database_path, 'w') as file:
            json.dump(data, file, indent=2)

    def read_or_initiate_database(self):
        if not os.path.isfile(self.client_database_path):
            self.write_database({
                'client_uid': str(uuid.uuid4()),
                'client_logs': []
            })

        with open(self.client_database_path, 'r') as file:
            data = json.load(file)
        
        return data
    
    def make_backup(self):
        self.write_database({
            'client_uid': self.client_uid,
            'client_logs': self.client_logs
        })
        
    def fit(
        self,  
        parameters: List[np.ndarray], 
        config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict]:
        while True:
            try:
                # Load data
                trainloader, testloader = NN.load_client_data(self.node_id)

                # Set model parameters, train model, return updated model parameters
                model = NN.load_model().to(self.device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0)

                set_parameters(model, parameters)

                NN.train(model, optimizer, trainloader, self.device, 1)
                loss, accuracy = NN.test(model, testloader, self.device)

                if NOISE_ABS_STD is not None:
                    add_constant_gaussian_noise_to_model(model, self.device, NOISE_ABS_STD)
                if NOISE_PERCENTAGE_STD is not None:
                    add_percentage_gaussian_noise_to_model(model, self.device, NOISE_PERCENTAGE_STD)

                model_params = get_parameters(model)
                if PARAMS_COMPRESSION_RATE is not None:
                    model_params = compress_parameters(model_params, PARAMS_COMPRESSION_RATE)

                trained_local_result = (model_params, len(trainloader.dataset))

                with open(self.result_cach_path, 'wb') as file:
                    pickle.dump((trained_local_result, loss, accuracy), file)

                # break

                del model, trainloader, testloader, trained_local_result, loss, accuracy

                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                return 
            except Exception as e:
                print(f"Training failed {e}")
                print(f"Sleeping for random seconds before retrying")
                time.sleep(random.randrange(5, 14))

                # with open(RESULT_CACHE_PATH, 'w') as file:
                #     file.write("FAILED")

    def spawn_new_local_training(self, epoch, parameters):
        # self.discard_local_training()
        
        self.local_traing_process = multiprocessing.Process(target=self.fit, args=(parameters, {}))

        if os.path.exists(self.result_cach_path):
            os.remove(self.result_cach_path)

        self.current_training_epoch = epoch
        self.local_traing_process.start()
        self.local_training_in_progress = True
    
    def load_local_training_result_if_done(self):
        if not os.path.exists(self.result_cach_path) or self.local_traing_process.exitcode != 0:
            print("Training failed")
            return False
        
        with open(self.result_cach_path, 'rb') as file:
            self.local_model_result, self.local_model_loss, self.local_model_accuracy = pickle.load(file)

        self.client_logs.append({
            'epoch': self.current_training_epoch,
            'status': 'TRAINING_COMPLETED',
            'loss': self.local_model_loss,
            'accuracy': self.local_model_accuracy
        })

        self.make_backup()
        return True

def write_global_log(global_model_epoch, global_loss, global_accuracy, epoch_start_time):
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
        'training_time': time.time() - epoch_start_time
    })

    with open(SERVER_LOG_DATABASE_PATH, 'w') as file:
        json.dump(data, file, indent=2)

def write_global_model(file_path, global_model_epoch, global_model_params, global_loss, global_accuracy):

    with open(file_path, 'wb') as file:
        pickle.dump({
            'global_model_epoch': global_model_epoch,
            'global_model_params': global_model_params,
            'global_loss': global_loss,
            'global_accuracy': global_accuracy,
        }, file)

def read_global_model(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    return data

def read_or_initialize_global_model():
    if not os.path.isfile(PERMANENT_GLOBAL_MODEL_PATH):
        write_global_model(
            PERMANENT_GLOBAL_MODEL_PATH, -1, get_parameters(NN.load_model().to(CPU_DEVICE)), None, None
        )
    
    return read_global_model(PERMANENT_GLOBAL_MODEL_PATH)

def centralized_aggregation(test_loader, current_training_epoch, client_results):
    while True:
        try:
            global_model_params = fedavg_aggregate(client_results)

            global_model = NN.load_model().to(DEVICE)
            set_parameters(global_model, global_model_params)

            global_loss, global_accuracy = NN.test(global_model, test_loader, DEVICE)

            print(f"Aggregated result: Device: {DEVICE}, Training epoch {current_training_epoch}, Loss {global_loss}, Acc {global_accuracy}")

            if NOISE_ABS_STD is not None:
                add_constant_gaussian_noise_to_model(global_model, DEVICE, NOISE_ABS_STD)
            if NOISE_PERCENTAGE_STD is not None:
                add_percentage_gaussian_noise_to_model(global_model, DEVICE, NOISE_PERCENTAGE_STD)

            global_model_params = get_parameters(global_model)
            if PARAMS_COMPRESSION_RATE is not None:
                global_model_params = compress_parameters(global_model_params, PARAMS_COMPRESSION_RATE)

            write_global_model(PERMANENT_GLOBAL_MODEL_PATH, current_training_epoch, global_model_params, global_loss, global_accuracy)

            
            return global_model_params, global_loss, global_accuracy
        except Exception as e:
            print(f"Error in aggregation: {e}")
            time.sleep(5)

NUM_EPOCHS = 100 if os.getenv('NUM_EPOCHS') is None else int(os.getenv('NUM_EPOCHS'))
NUM_CLIENTS = 8
MIN_FIT_CLIENTS = 8 if os.getenv('MIN_FIT_CLIENTS') is None else int(os.getenv('MIN_FIT_CLIENTS'))


if __name__ == "__main__":
    test_loader = NN.load_test_data()
    clients = [Client(node_id) for node_id in range(NUM_CLIENTS)]

    multiprocessing.set_start_method('forkserver')

    # initialize global model
    global_model_data = read_or_initialize_global_model()
    global_model_epoch = global_model_data['global_model_epoch']

    global_model_params = global_model_data['global_model_params']

    # while global_model_epoch < NUM_EPOCHS:
    while True:
        epoch_start_time = time.time()
        global_model_epoch += 1
        for node_id in range(NUM_CLIENTS):
            clients[node_id].spawn_new_local_training(global_model_epoch, global_model_params)

        for node_id in range(NUM_CLIENTS):
            clients[node_id].local_traing_process.join()

        for node_id in range(NUM_CLIENTS):
            clients[node_id].load_local_training_result_if_done()

        client_results = []
        for node_id in range(NUM_CLIENTS):
            client_results.append(clients[node_id].local_model_result)

        global_model_params, global_loss, global_accuracy = centralized_aggregation(test_loader, global_model_epoch, client_results)

        write_global_log(global_model_epoch, global_loss, global_accuracy, epoch_start_time)
        