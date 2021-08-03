import logging
import torch
import numpy as np
from rpc import RL_pb2_grpc
from rpc.RL_pb2 import Prediction, Empty
from collections import namedtuple
from queue import Queue
from threading import Lock, Event, Thread
from config import N_PLAYERS, MAX_INFERENCE_BATCH_SIZE, OBS_SHAPE, DATA_PROCESS_TIMEOUT, SEQUENCE_LENGTH
from networks import RegretNetwork, StrategyNetwork

Observations = namedtuple("Observations", "obs count")
REGRET = 0
STRATEGY = 1
types = ['regret', 'strategy']


class Actor(RL_pb2_grpc.ActorServicer):
    def __init__(self, gpu_lock):
        self.gpu_lock = gpu_lock
        # Containers for all data required for regret and strategy inference
        self.player_locks = [[Lock() for _ in range(N_PLAYERS)] for type in types]
        self.current_batch_ids = [np.zeros(N_PLAYERS, dtype=np.int32) for type in types]
        self.current_batch_sizes = [[[0] for _ in range(N_PLAYERS)] for type in types]
        self.tmp_batch_obs = [[np.zeros([MAX_INFERENCE_BATCH_SIZE, SEQUENCE_LENGTH, OBS_SHAPE]) for _ in range(N_PLAYERS)] for type in types]
        self.tmp_batch_obs_count = [[np.zeros([MAX_INFERENCE_BATCH_SIZE]) for _ in range(N_PLAYERS)] for type in types]
        self.observations_que = [[Queue() for _ in range(N_PLAYERS)] for type in types]
        self.data_added_events = [[[Event()] for _ in range(N_PLAYERS)] for type in types]
        self.batch_full_events = [[[Event()] for _ in range(N_PLAYERS)] for type in types]
        self.batch_ready_events = [[[Event()] for _ in range(N_PLAYERS)] for type in types]
        self.batch_read_counts = [[[0] for _ in range(N_PLAYERS)] for type in types]
        self.action_predictions = [[dict() for _ in range(N_PLAYERS)] for type in types]
        self.bet_predictions = [[dict() for _ in range(N_PLAYERS)] for type in types]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logging.info('Pytorch using device %s' % self.device)
        self.regret_net = RegretNetwork(self.device).to(self.device)
        self.strategy_net = StrategyNetwork(self.device).to(self.device)
        # Which version of the strategy network is used for inference (for each player)
        self.strategy_versions = np.zeros(N_PLAYERS)
        for i in range(N_PLAYERS):
            Thread(target=self._process_batch_thread, args=(i, REGRET)).start()
            Thread(target=self._process_batch_thread, args=(i, STRATEGY)).start()

    def _finalize_batch(self, type, player):
        current_batch = self.current_batch_ids[type][player]
        current_batch_size = self.current_batch_sizes[type][player][current_batch]
        self.observations_que[type][player].put((self.tmp_batch_obs[type][player][:current_batch_size], self.tmp_batch_obs_count[type][player][:current_batch_size]))
        self.tmp_batch_obs[type][player] = np.zeros([MAX_INFERENCE_BATCH_SIZE, SEQUENCE_LENGTH, OBS_SHAPE])
        self.tmp_batch_obs_count[type][player] = np.zeros([MAX_INFERENCE_BATCH_SIZE])
        self.current_batch_ids[type][player] += 1
        # Add required entries for the next batch
        self.current_batch_sizes[type][player].append(0)
        self.batch_read_counts[type][player].append(0)
        self.batch_ready_events[type][player].append(Event())
        self.batch_full_events[type][player].append(Event())
        self.data_added_events[type][player].append(Event())

    def _process_batch_thread(self, player, type):
        logging.info("Started %s inference data processing thread for player %d" % (types[type], player))
        current_batch = 0
        while True:
            self.data_added_events[type][player][current_batch].wait(timeout=None)
            # Once any amount of data has been added to the current batch, wait until the batch is full, or until the specified timeout
            self.batch_full_events[type][player][current_batch].wait(timeout=DATA_PROCESS_TIMEOUT)
            self.player_locks[type][player].acquire()
            this_batch_size = self.current_batch_sizes[type][player][current_batch]
            # If we arrived here by timing out above, we need to set the current batch as consumed, so that we no longer add data to it
            if this_batch_size != MAX_INFERENCE_BATCH_SIZE:
                self._finalize_batch(type, player)
            self.player_locks[type][player].release()
            logging.debug("Processing %s batch size of: %d, for player: %d" % (types[type], this_batch_size, player))
            observations, counts = self.observations_que[type][player].get()
            self.gpu_lock.acquire()
            observations = torch.from_numpy(observations[:this_batch_size]).type(torch.FloatTensor).to(self.device)
            counts = torch.from_numpy(counts[:this_batch_size]).type(torch.LongTensor)
            if type == REGRET:
                self.regret_net.load_state_dict(torch.load('../states/regret_net_player_%d' % player))
                action_predictions, bet_predictions = self.regret_net(observations, counts)
            else:
                self.strategy_net.load_state_dict(torch.load('../states/strategy_net_%d' % self.strategy_versions[player]))
                action_predictions, bet_predictions = self.strategy_net(observations, counts)
            self.gpu_lock.release()
            self.player_locks[type][player].acquire()
            self.action_predictions[type][player][current_batch] = action_predictions.detach().cpu().numpy()
            self.bet_predictions[type][player][current_batch] = bet_predictions.detach().cpu().numpy()
            self.batch_ready_events[type][player][current_batch].set()
            self.player_locks[type][player].release()
            current_batch += 1

    def _add_observations_to_batch(self, player, observations, counts, type):
        self.player_locks[type][player].acquire()
        batch_ids = []
        indices = []
        current_batch = self.current_batch_ids[type][player]
        if len(observations) + self.current_batch_sizes[type][player][current_batch] >= MAX_INFERENCE_BATCH_SIZE:
            consumed = 0
            remaining = len(observations)
            current_batch_size = self.current_batch_sizes[type][player][current_batch]
            space_left = MAX_INFERENCE_BATCH_SIZE - current_batch_size
            while remaining >= space_left:
                batch_ids.append(self.current_batch_ids[type][player])
                indices.append([*range(self.current_batch_sizes[type][player][current_batch], MAX_INFERENCE_BATCH_SIZE)])
                self.current_batch_sizes[type][player][current_batch] = MAX_INFERENCE_BATCH_SIZE
                self.tmp_batch_obs[type][player][current_batch_size:MAX_INFERENCE_BATCH_SIZE](observations[consumed:consumed + space_left])
                self.tmp_batch_obs_count[type][player][current_batch_size:MAX_INFERENCE_BATCH_SIZE].append(counts[consumed:consumed + space_left])
                self._finalize_batch(type, player)
                self.data_added_events[type][player][current_batch].set()
                self.batch_full_events[type][player][current_batch].set()
                current_batch_size = 0
                current_batch += 1
                consumed += space_left
                space_left = MAX_INFERENCE_BATCH_SIZE
                remaining = len(observations) - consumed
            if remaining > 0:
                batch_ids.append(self.current_batch_ids[type][player])
                indices.append([*range(0, remaining)])
                self.current_batch_sizes[type][player][current_batch] = remaining
                self.tmp_batch_obs[type][player][0:remaining] = observations[consumed:consumed + remaining]
                self.tmp_batch_obs_count[type][player][0:remaining] = counts[consumed:consumed + remaining]
                self.data_added_events[type][player][current_batch].set()
        else:
            current_batch = self.current_batch_ids[type][player]
            batch_ids.append(current_batch)
            current_batch_size = self.current_batch_sizes[type][player][current_batch]
            indices.append([*range(current_batch_size, current_batch_size+len(observations))])
            self.tmp_batch_obs[type][player][current_batch_size:current_batch_size+len(observations)] = observations
            self.tmp_batch_obs_count[type][player][current_batch_size:current_batch_size+len(observations)] = counts
            self.current_batch_sizes[type][player][current_batch] += len(observations)
            self.data_added_events[type][player][current_batch].set()
        self.player_locks[type][player].release()
        return batch_ids, indices

    def _retrieve_batch_result(self, player, batch_id, indices, type):
        self.player_locks[type][player].acquire()
        self.batch_read_counts[type][player][batch_id] += len(indices)
        action_regrets = self.action_predictions[type][player][batch_id][indices]
        bet_regrets = self.bet_predictions[type][player][batch_id][indices]
        if self.batch_read_counts[type][player][batch_id] == self.current_batch_sizes[type][player][batch_id]:
            self.action_predictions[type][player].pop(batch_id)
            self.bet_predictions[type][player].pop(batch_id)
        self.player_locks[type][player].release()
        return action_regrets, bet_regrets

    def GetRegrets(self, request, context):
        player = int(request.player)
        observations = np.frombuffer(request.observations, dtype=np.float32).reshape(request.shape, request.sequence_length, OBS_SHAPE)
        obs_counts = np.frombuffer(request.observation_counts, dtype=np.int32).reshape(request.shape)
        batch_ids, indices = self._add_observations_to_batch(player, observations, obs_counts, REGRET)
        action_regrets = []
        bet_regrets = []
        for batch_id, indices in zip(batch_ids, indices):
            self.batch_ready_events[REGRET][player][batch_id].wait(timeout=None)
            batch_action_regrets, batch_bet_regrets = self._retrieve_batch_result(player, batch_id, indices, REGRET)
            action_regrets.extend(batch_action_regrets)
            bet_regrets.extend(batch_bet_regrets)
        action_bytes = np.ndarray.tobytes(np.asarray(action_regrets))
        bet_bytes = np.ndarray.tobytes(np.asarray(bet_regrets))
        return Prediction(action_prediction=action_bytes, bet_prediction=bet_bytes)

    def GetStrategies(self, request, context):
        player = int(request.player)
        observations = np.frombuffer(request.observations, dtype=np.float32).reshape(request.shape, request.sequence_length, OBS_SHAPE)
        obs_counts = np.frombuffer(request.observation_counts, dtype=np.int32).reshape(request.shape)
        batch_ids, indices = self._add_observations_to_batch(player, observations, obs_counts, STRATEGY)
        action_regrets = []
        bet_regrets = []
        for batch_id, indices in zip(batch_ids, indices):
            self.batch_ready_events[STRATEGY][player][batch_id].wait(timeout=None)
            batch_action_regrets, batch_bet_regrets = self._retrieve_batch_result(player, batch_id, indices, STRATEGY)
            action_regrets.extend(batch_action_regrets)
            bet_regrets.extend(batch_bet_regrets)
        action_bytes = np.ndarray.tobytes(np.asarray(action_regrets))
        bet_bytes = np.ndarray.tobytes(np.asarray(bet_regrets))
        return Prediction(action_prediction=action_bytes, bet_prediction=bet_bytes)

    def SetStrategy(self, request, context):
        player = int(request.player)
        version = int(request.strategy_version)
        self.strategy_versions[player] = version
        return Empty()

