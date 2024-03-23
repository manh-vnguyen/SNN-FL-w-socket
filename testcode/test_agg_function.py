import torch
import pickle

from typing import Dict, List, Optional, Tuple, Any
from collections import OrderedDict
import numpy as np
import numpy.typing as npt
from functools import reduce

NDArray = npt.NDArray[Any]
NDArrayInt = npt.NDArray[np.int_]
NDArrayFloat = npt.NDArray[np.float_]
NDArrays = List[NDArray]

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

with open('database/server_database.pkl', 'rb') as file:
    data = pickle.load(file)

    
client_model_record = data['client_model_record']

print(client_model_record.shape)

