import numpy as np
from pokerenv.table import Table

# -------------------------------------- GENERAL ---------------------------------------
N_PLAYERS = 2
N_ITERATIONS = 25
K = 10

# --------------------------------------- WORKERS ----------------------------------------
MASTER_HOST = 'localhost:50040'
SLAVE_HOSTS = [
    'localhost:50041'
]
GLOBAL_STRATEGY_HOST = 'localhost:50050'
GLOBAL_EVAL_HOST = 'localhost:50070'
ACTOR_HOST_PLAYER_MAP = {
    'localhost:50051': [0, 1],
}
REGRET_HOST_PLAYER_MAP = {
    'localhost:50061': [0, 1],
}
PLAYER_ACTOR_HOST_MAP = {k: new_values for new_values, new_keys in zip(ACTOR_HOST_PLAYER_MAP.keys(), ACTOR_HOST_PLAYER_MAP.values()) for k in new_keys}
PLAYER_REGRET_HOST_MAP = {k: new_values for new_values, new_keys in zip(REGRET_HOST_PLAYER_MAP.keys(), REGRET_HOST_PLAYER_MAP.values()) for k in new_keys}


# -------------------------------- LEARNING ENVIRONMENT --------------------------------
env = Table(6)
OBS_SHAPE = env.observation_space.shape[0]
N_ACTIONS = env.action_space.spaces[0].n
RANDOM_SEED = 1
LOW_STACK_BBS = 50
HIGH_STACK_BBS = 200
INVALID_ACTION_PENALTY = 0


# --------------------------------------- GLOBAL -----------------------------------------
N_DEFAULT_BET_BUCKETS = 3                                        # Default bet buckets are min bet, max bet and (min bet + max bet) / 2
N_USER_BET_BUCKETS = 5                                           # How many other bet sizes are considered by the regret/strategy networks
N_BET_BUCKETS = N_DEFAULT_BET_BUCKETS + N_USER_BET_BUCKETS
BET_BUCKETS = np.array([0.75, 1, 1.5, 2, 3])                     # Which (additional) bet sizes are considered (specified as % of the pot)
MAX_EPISODE_LENGTH = 25
SEQUENCE_LENGTH = 5                                              # How many previous observations are fed to RNN networks

# ----------------------------------------- DCRM -----------------------------------------
N_TRAVERSE_PROCESSES = 5
N_TRAVERSE_QUE_PROCESS = 2
N_CONC_TRAVERSALS_PER_PROCESS = 1
CLIENT_SAMPLES_MIN_BATCH_SIZE = 512
EXTERNAL_SAMPLING = True
OUTCOME_SAMPLING_EPSILON = 0.6


# ------------------------------------- EVAL GENERAL -------------------------------------
N_LOGGED_HANDS = 100                                             # How many hand histories are recorded per eval iteration
USE_PPO_EVALUATION = False
STUPID_MCTS = True
EVAL_CONSIDER_SINGLE_TRAJECTORY = True                           # Consider only a single set of table cards for each evaluation hand


# ----------------------------------------- PPO ------------------------------------------
N_PPO_EVAL_PROCESSES = 6
N_PPO_EVAL_QUE_PROCESSES = 1
N_CONC_PPO_EVALUATIONS_PER_PROCESS = 512
PPO_EVAL_TRAJECTORY_SAMPLES_MIN_BATCH_SIZE = 10*1024
PPO_EVAL_REWARD_SAMPLES_MIN_BATCH_SIZE = 512
N_EVAL_ITERATIONS = 500                                          # How many different hands are considered
N_PPO_TRAINING_HANDS = 50*1000                                   # How many times each hand is trained
N_PPO_EVAL_HANDS = 5*1000                                        # How many times each hand is evaluated
DISCOUNT_RATE = 0.95                                             # Discount rate for rewards
EVAL_PERMISSION_WAIT_TIME = 1                                    # How long to wait for before checking for permission to run evaluations

# ------------------------------ APPROXIMATE EXPLOITABILITY ------------------------------
N_MCTS_PROCESSES = 6
INITIAL_VALUE = 1                                                # What value each state is initialized to
PB_C_BASE = 19652
PB_C_INIT = 1.25
DIRICHLET_ALPHA = 0.3
EXPLORATION_FRACTION = 0.25
N_MCTS_EVAL_HANDS = 5
N_MONTE_CARLO_SIMS = 200

# ---------------------------------------- SERVER ----------------------------------------
N_THREADPOOL_WORKERS = 8
LINEAR_CFR = True
SINGLE_NETWORK = False

# DCRM
DCRM_RESERVOIR_SIZE = int(5e6)                                   # How many samples are stored in each reservoir
DCRM_BATCH_PROCESS_TIMEOUT = 0.005                               # Timeout duration before a batch is processed even if it is not full
DCRM_MAX_INFERENCE_BATCH_SIZE = 1024*5                           # Batch size for inferring regrets or strategies
DCRM_MAX_TRAIN_BATCH_SIZE = 1024*5                               # Batch size when training networks

# MCTS
MCTS_RESERVOIR_SIZE = int(1e6)
MCTS_MAX_INFERENCE_BATCH_SIZE = 1024*5                           # Batch size for inferring regrets or strategies
MCTS_MIN_TRAIN_BATCH_SIZE = 1024*10                              # Min fresh samples before triggering training
MCTS_MAX_TRAIN_BATCH_SIZE = 1024*10                              # Batch size when training networks
MCTS_BATCH_PROCESS_TIMEOUT = 0.5                                 # Timeout duration before a batch is processed even if it is not full
MCTS_TRAINING_TIMEOUT = 30                                       # How long to wait after data is added, before training

# PPO
PPO_EXPERIENCE_BUFFER_SIZE = int(1e5)
PPO_MAX_INFERENCE_BATCH_SIZE = 1024*5                           # Batch size for inferring regrets or strategies
PPO_MAX_TRAIN_BATCH_SIZE = 1024*5                               # Batch size when training networks
PPO_BATCH_PROCESS_TIMEOUT = 0.5                                 # Timeout duration before a batch is processed even if it is not full
PPO_TRAINING_TIMEOUT = 5
PPO_CLIP_VALUE = 0.1


# --------------------------------- NETWORKS & TRAINING ----------------------------------
PATIENCE = 10
improvement_eps = 1.05

# DCRM
N_DCRM_EPOCHS = 50
DCRM_RNN_HIDDENS = 2048
REGRET_LEARNING_RATE = 0.001
STRATEGY_LEARNING_RATE = 0.001
REGRET_WEIGHT_DECAY = 0.001
STRATEGY_WEIGHT_DECAY = 0.001

# MCTS
N_MCTS_EPOCHS = 50
MCTS_RNN_HIDDENS = 1024
VALUE_P_LEARNING_RATE = 0.001
VALUE_P_WEIGHT_DECAY = 0.001

# PPO
N_PPO_EPOCHS = 50
PPO_RNN_HIDDENS = 1024
ACTOR_LEARNING_RATE = 0.0004
ACTOR_WEIGHT_DECAY = 0.0001
CRITIC_LEARNING_RATE = 0.0001
CRITIC_WEIGHT_DECAY = 0.0001


