import numpy as np
from pokerenv.table import Table

# --------------------------------------- WORKERS ----------------------------------------
SERVER_HOST = ['localhost:5050', 'localhost:5051']
CLIENT_HOSTS = [
    'localhost:5053'
]


# -------------------------------------- GENERAL ---------------------------------------
N_ITERATIONS = 25
K = 1500
TRAVERSALS_PER_PROCESS = 1
N_PROCESSES_PER_WORKER = 5
assert K % (len(CLIENT_HOSTS) * TRAVERSALS_PER_PROCESS * N_PROCESSES_PER_WORKER) == 0
K_PER_WORKER = K / (len(CLIENT_HOSTS) * TRAVERSALS_PER_PROCESS * N_PROCESSES_PER_WORKER)
N_EVALUATIONS = 5e4
N_LATEST_POLICIES = 3                                            # How many previous policies the current policy is evaluated against


# -------------------------------- LEARNING ENVIRONMENT --------------------------------
N_PLAYERS = 2
RANDOM_SEED = 1
LOW_STACK_BBS = 50
HIGH_STACK_BBS = 200
HH_LOCATION = '../hands/'
INVALID_ACTION_PENALTY = 0


# --------------------------------------- GLOBAL -----------------------------------------
env = Table(6)
OBS_SHAPE = env.observation_space.shape[0]
SEQUENCE_LENGTH = 5
N_ACTIONS = env.action_space.spaces[0].n
N_DEFAULT_BET_BUCKETS = 3                                        # Default bet buckets are min bet, max bet and (min bet + max bet) / 2
N_USER_BET_BUCKETS = 4                                           # How many other bet sizes are considered by the regret/strategy networks
N_BET_BUCKETS = N_DEFAULT_BET_BUCKETS + N_USER_BET_BUCKETS
BET_BUCKETS = np.array([0.75, 1, 1.5, 2])                        # Which (additional) bet sizes are considered (specified as % of the pot)


# ---------------------------------------- CLIENT ----------------------------------------
N_TRAVERSE_PROCESSES = 5
N_CONC_TRAVERSALS_PER_PROCESS = 1
N_QUE_PROCESS = 3
CLIENT_SAMPLES_MIN_BATCH_SIZE = 1024                                 # Batch size for sampled regrets
EVAL_HH_FREQUENCY = 10000
EVAL_ENVS_PER_PROCESS = 1000
N_EVAL_HANDS = 100000
N_EVAL_PROCESSES = 5


# ---------------------------------------- SERVER ----------------------------------------
RESERVOIR_SIZE = int(5e6)                                        # How many samples are stored in each reservoir
DATA_PROCESS_TIMEOUT = 0.005                                     # Timeout duration before a batch is processed even if it is not full
MAX_INFERENCE_BATCH_SIZE = 1024*10                               # Batch size for inferring regrets or strategies
MAX_TRAIN_BATCH_SIZE = 1024*10                                   # Batch size when training networks


# --------------------------------- NETWORKS & TRAINING ----------------------------------
RNN_HIDDENS = 2048
N_EPOCHS = 50
REGRET_LEARNING_RATE = 0.01
STRATEGY_LEARNING_RATE = 0.01
REGRET_WEIGHT_DECAY = 0.001
STRATEGY_WEIGHT_DECAY = 0.001
PATIENCE = 25
improvement_eps = 1.05

