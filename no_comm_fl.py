
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

parser = argparse.ArgumentParser(description="Flower")
parser.add_argument("--db_postfix", type=str, default="")
ARGS = parser.parse_args()

load_dotenv(f".env_no_comm")

if os.getenv('MODEL') == 'SNN':
    import cifar10_SNN as cifar10
elif os.getenv('MODEL') == 'ANN':
    import cifar10_ANN as cifar10
else:
    raise Exception("Model type wrong!!")

NOISE_ABS_STD =  None if os.getenv('NOISE').split(',')[0] == '_' else float(os.getenv('NOISE').split(',')[0])
NOISE_PERCENTAGE_STD = None if os.getenv('NOISE').split(',')[1] == '_' else float(os.getenv('NOISE').split(',')[1])

from fedlearn import sha256_hash, set_parameters, get_parameters, add_percentage_gaussian_noise_to_model, add_constant_gaussian_noise_to_model

gpu_assignment = [int(x) for x in os.getenv('GPU_ASSIGNMENT').split(',')]

NUM_EPOCHS = 10

class Client:
    def __init__(self, node_id) -> None:
        self.node_id = node_id
        self.device = torch.device(f"cuda:{gpu_assignment[self.node_id]}" if torch.cuda.is_available() else "cpu")
        self.result_cach_path = f"database{ARGS.db_postfix}/training_result_{self.node_id}.pkl"
        
    def fit(
        self,  
        parameters: List[np.ndarray], 
        config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict]:
        while True:
            try:
                # Load data
                trainloader, testloader = cifar10.load_client_data(ARGS.node_id)

                # Set model parameters, train model, return updated model parameters
                model = cifar10.load_model().to(self.device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0)

                set_parameters(model, parameters)

                cifar10.train(model, optimizer, trainloader, self.device, 1)
                loss, accuracy = cifar10.test(model, testloader, self.device)

                if NOISE_ABS_STD is not None:
                    add_constant_gaussian_noise_to_model(model, self.device, NOISE_ABS_STD)
                if NOISE_PERCENTAGE_STD is not None:
                    add_percentage_gaussian_noise_to_model(model, self.device, NOISE_PERCENTAGE_STD)

                trained_local_result = (get_parameters(model), len(trainloader.dataset))

                with open(self.result_cach_path, 'wb') as file:
                    pickle.dump((trained_local_result, loss, accuracy), file)

                # break

                del model, trainloader, testloader, trained_local_result, loss, accuracy

                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                return 
            except Exception as e:
                print(f"Training failed {e}")
                random.randrange(5, 14)
                print(f"Sleeping for random seconds before retrying")

                # with open(RESULT_CACHE_PATH, 'w') as file:
                #     file.write("FAILED")

    def spawn_new_local_training(self, epoch, parameters):
        self.discard_local_training()
        
        self.local_traing_process = multiprocessing.Process(target=self.fit, args=(parameters, {}))
        # self.local_traing_process = threading.Thread(target=self.fit, args=(parameters, {}))

        if os.path.exists(RESULT_CACHE_PATH):
            os.remove(RESULT_CACHE_PATH)

        self.current_training_epoch = epoch
        self.local_traing_process.start()
        self.local_training_in_progress = True
    
    def load_local_training_result_if_done(self):
        if not os.path.exists(RESULT_CACHE_PATH) or self.is_running_local_training():
            return False
        
        with open(RESULT_CACHE_PATH, 'rb') as file:
            self.local_model_result, self.local_model_loss, self.local_model_accuracy = pickle.load(file)

        self.local_training_in_progress = False

        print(json.dumps(self.client_logs[-1], indent=2))
        return True

for epoch in NUM_EPOCHS:
