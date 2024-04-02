import os

gpu_assignment = '0,0,0,0,0,0,0,0,0,0'
server_gpu_assignment = 0
noise_dot = 6

test_name = f"CIFAR100_IID"
db_folder = f"database_ANN_cifar10_IID_client_10o10_noise_p_dot{noise_dot}"


ENV_STRING = f"""
MODEL=TSP_ANN

NUM_STEPS=_
BATCH_SIZE=32

LR_INITIAL=0.001
LR_INTERVALS=20,30
LR_REDUCTION=10

OPT_MOMENTUM=_
WEIGHT_DECAY=5e-4

NOISE=_,0.{noise_dot}
PARAMS_COMPRESSION_RATE=_

GPU_ASSIGNMENT={gpu_assignment}
SERVER_GPU_ASSIGNMENT={server_gpu_assignment}

DATA_PATH=/tmp/data/CIFAR10-IID-10-CLIENT.pkl
NUM_OUTPUTS=10


NUM_CLIENTS=10
MIN_FIT_CLIENTS=10
NUM_EPOCHS=40
NUM_LOCAL_EPOCHS=5


# tmux new-session -d -s ANN_cifar10_IID_client_10o10_noise_p_dot{noise_dot}
# tmux send-keys -t ANN_cifar10_IID_client_10o10_noise_p_dot{noise_dot} "python3 no_comm_fl.py --db_postfix _ANN_cifar10_IID_client_10o10_noise_p_dot{noise_dot}" C-m
# tmux kill-session -t ANN_cifar10_IID_client_10o10_noise_p_dot{noise_dot}
"""

if not os.path.exists(db_folder):
    os.mkdir(db_folder)

with open(f"{db_folder}/.env", "w") as file:
    file.write(ENV_STRING)

print(ENV_STRING)

print(f"""
tmux new-session -d -s ANN_cifar10_IID_client_10o10_noise_p_dot{noise_dot}
tmux send-keys -t ANN_cifar10_IID_client_10o10_noise_p_dot{noise_dot} "python3 no_comm_fl.py --db_postfix _ANN_cifar10_IID_client_10o10_noise_p_dot{noise_dot}" C-m
""")