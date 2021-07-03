import numpy as np

# -------------------------------- LEARNING ENVIRONMENT --------------------------------
RANDOM_SEED = 1
N_PLAYERS = 2
LOW_STACK_BBS = 50
HIGH_STACK_BBS = 200
HH_LOCATION = 'hands/'
INVALID_ACTION_PENALTY = 0


# --------------------------------------- GLOBAL ----------------------------------------
OBS_SHAPE = 60
SEQUENCE_LENGTH = 5
N_ACTIONS = 4
N_DEFAULT_BET_BUCKETS = 3                                        # Default bet buckets are min bet, max bet and (min bet + max bet) / 2
N_USER_BET_BUCKETS = 5                                           # How many bet sizes are considered by the regret/strategy networks
N_BET_BUCKETS = N_DEFAULT_BET_BUCKETS + N_USER_BET_BUCKETS
BET_BUCKETS = np.array([0.5, 0.75, 1, 1.5, 3])                   # Which bet sizes are considered (specified as % of the pot)


# ---------------------------------------- CLIENT ----------------------------------------
CLIENT_SAMPLES_BATCH_SIZE = 1024*2                               # Batch size for sampled regrets


# ---------------------------------------- SERVER ----------------------------------------
RESERVOIR_SIZE = int(1e6)                                        # How many samples are stored in each reservoir
DATA_PROCESS_TIMEOUT = 0.005                                     # Timeout duration before a batch is processed even if it is not full
MAX_INFERENCE_BATCH_SIZE = 1024*10                               # Batch size for inferring regrets or strategies
MAX_TRAIN_BATCH_SIZE = 1024*10                                   # Batch size when training networks
STATES_LOCATION = ''
RESERVOIRS_LOCATION = ''


# --------------------------------- NETWORKS & TRAINING ----------------------------------
RNN_HIDDENS = 2048
N_EPOCHS = 1000
LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.001
PATIENCE = 50
improvement_eps = 1.05
