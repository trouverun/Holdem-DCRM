import logging
import torch
import numpy as np
from rpc import RL_pb2_grpc
from queue import Queue
from server.util import BatchManager
from rpc.RL_pb2 import Prediction, Empty
from threading import Lock, Thread
from server.networks import RegretNetwork, PolicyNetwork
from config import N_PLAYERS, DCRM_MAX_INFERENCE_BATCH_SIZE, OBS_SHAPE, DCRM_BATCH_PROCESS_TIMEOUT


class Actor(RL_pb2_grpc.ActorServicer):
    def __init__(self, player_list, gpu_lock):
        self.gpu_lock = gpu_lock
        self.player_regret_locks = {player: Lock() for player in player_list}
        self.player_strategy_locks = {player: Lock() for player in player_list}
        self.regret_batch_que = {player: Queue() for player in player_list}
        self.strategy_batch_que = {player: Queue() for player in player_list}
        self.regret_batch_managers = {player: BatchManager(DCRM_MAX_INFERENCE_BATCH_SIZE, self.regret_batch_que[player]) for player in player_list}
        self.strategy_batch_managers = {player: BatchManager(DCRM_MAX_INFERENCE_BATCH_SIZE, self.strategy_batch_que[player]) for player in player_list}

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logging.info('Pytorch using device %s' % self.device)
        self.regret_net = RegretNetwork(self.device).to(self.device)
        self.strategy_net = PolicyNetwork(self.device).to(self.device)
        # Which version of the strategy network is used for inference (for each player)
        self.strategy_versions = np.zeros(N_PLAYERS)
        for player in player_list:
            Thread(target=self._process_regret_batch_thread, args=(player,)).start()
            Thread(target=self._process_strategy_batch_thread, args=(player,)).start()

    def _process_regret_batch_thread(self, player):
        logging.info("Started regret inference data processing thread for player %d" % player)
        current_batch = 0
        while True:
            self.regret_batch_managers[player].data_added_events[current_batch].wait(timeout=None)
            # Once any amount of data has been added to the current batch, wait until the batch is full, or until the specified timeout
            self.regret_batch_managers[player].batch_full_events[current_batch].wait(timeout=DCRM_BATCH_PROCESS_TIMEOUT)
            self.player_regret_locks[player].acquire()
            this_batch_size = self.regret_batch_managers[player].current_batch_sizes[current_batch]
            # If we arrived here by timing out above, we need to set the current batch as consumed, so that we no longer add data to it
            if this_batch_size != DCRM_MAX_INFERENCE_BATCH_SIZE:
                self.regret_batch_managers[player].finalize_batch()
            self.player_regret_locks[player].release()
            logging.debug("Processing regret batch size of: %d, for player: %d" % (this_batch_size, player))
            observations, counts = self.regret_batch_que[player].get()
            self.gpu_lock.acquire()
            #self.regret_net.to(self.device)
            observations = torch.from_numpy(observations[:this_batch_size]).type(torch.FloatTensor).to(self.device)
            counts = torch.from_numpy(counts[:this_batch_size]).type(torch.LongTensor)
            self.regret_net.load_state_dict(torch.load('states/regret/regret_net_player_%d' % player))
            with torch.no_grad():
                action_predictions, bet_predictions = self.regret_net(observations, counts)
            self.player_regret_locks[player].acquire()
            self.regret_batch_managers[player].add_batch_results(current_batch,
                                                                 action_predictions.detach().cpu().numpy(), bet_predictions.detach().cpu().numpy())
            self.player_regret_locks[player].release()
            #self.regret_net.to('cpu')
            torch.cuda.empty_cache()
            self.gpu_lock.release()
            current_batch += 1

    def _process_strategy_batch_thread(self, player):
        logging.info("Started strategy inference data processing thread for player %d" % player)
        current_batch = 0
        while True:
            self.strategy_batch_managers[player].data_added_events[current_batch].wait(timeout=None)
            # Once any amount of data has been added to the current batch, wait until the batch is full, or until the specified timeout
            self.strategy_batch_managers[player].batch_full_events[current_batch].wait(timeout=DCRM_BATCH_PROCESS_TIMEOUT)
            self.player_strategy_locks[player].acquire()
            this_batch_size = self.strategy_batch_managers[player].current_batch_sizes[current_batch]
            # If we arrived here by timing out above, we need to set the current batch as consumed, so that we no longer add data to it
            if this_batch_size != DCRM_MAX_INFERENCE_BATCH_SIZE:
                self.strategy_batch_managers[player].finalize_batch()
            self.player_strategy_locks[player].release()
            logging.debug("Processing strategy batch size of: %d, for player: %d" % (this_batch_size, player))
            observations, counts = self.strategy_batch_que[player].get()
            self.gpu_lock.acquire()
            #self.strategy_net.to(self.device)
            observations = torch.from_numpy(observations[:this_batch_size]).type(torch.FloatTensor).to(self.device)
            counts = torch.from_numpy(counts[:this_batch_size]).type(torch.LongTensor)
            self.strategy_net.load_state_dict(torch.load('states/strategy/strategy_net_%d' % self.strategy_versions[player]))
            with torch.no_grad():
                action_predictions, bet_predictions = self.strategy_net(observations, counts)
            self.player_strategy_locks[player].acquire()
            self.strategy_batch_managers[player].add_batch_results(current_batch, action_predictions.detach().cpu().numpy(), bet_predictions.detach().cpu().numpy())
            self.player_strategy_locks[player].release()
            #self.strategy_net.to('cpu')
            torch.cuda.empty_cache()
            self.gpu_lock.release()
            current_batch += 1

    def _add_regret_observations_to_batch(self, player, observations, counts):
        self.player_regret_locks[player].acquire()
        batch_ids, indices = self.regret_batch_managers[player].add_data_to_batch(observations, counts)
        self.player_regret_locks[player].release()
        return batch_ids, indices

    def _add_strategy_observations_to_batch(self, player, observations, counts):
        self.player_strategy_locks[player].acquire()
        batch_ids, indices = self.strategy_batch_managers[player].add_data_to_batch(observations, counts)
        self.player_strategy_locks[player].release()
        return batch_ids, indices

    def _retrieve_regret_batch_result(self, player, batch_id, indices):
        self.player_regret_locks[player].acquire()
        action_regrets, bet_regrets, _ = self.regret_batch_managers[player].get_batch_results(batch_id, indices)
        self.player_regret_locks[player].release()
        return action_regrets, bet_regrets

    def _retrieve_strategy_batch_result(self, player, batch_id, indices):
        self.player_strategy_locks[player].acquire()
        action_policy, bet_policy, _ = self.strategy_batch_managers[player].get_batch_results(batch_id, indices)
        self.player_strategy_locks[player].release()
        return action_policy, bet_policy

    def GetRegrets(self, request, context):
        player = int(request.player)
        observations = np.frombuffer(request.observations, dtype=np.float32).reshape(request.shape, request.sequence_length, OBS_SHAPE)
        obs_counts = np.frombuffer(request.observation_counts, dtype=np.int32).reshape(request.shape)
        batch_ids, indices = self._add_regret_observations_to_batch(player, observations, obs_counts)
        action_regrets = []
        bet_regrets = []
        for batch_id, indices in zip(batch_ids, indices):
            self.regret_batch_managers[player].batch_ready_events[batch_id].wait(timeout=None)
            batch_action_regrets, batch_bet_regrets = self._retrieve_regret_batch_result(player, batch_id, indices)
            action_regrets.extend(batch_action_regrets)
            bet_regrets.extend(batch_bet_regrets)
        action_bytes = np.ndarray.tobytes(np.asarray(action_regrets))
        bet_bytes = np.ndarray.tobytes(np.asarray(bet_regrets))
        return Prediction(action_prediction=action_bytes, bet_prediction=bet_bytes)

    def GetStrategies(self, request, context):
        player = int(request.player)
        observations = np.frombuffer(request.observations, dtype=np.float32).reshape(request.shape, request.sequence_length, OBS_SHAPE)
        obs_counts = np.frombuffer(request.observation_counts, dtype=np.int32).reshape(request.shape)
        batch_ids, indices = self._add_strategy_observations_to_batch(player, observations, obs_counts)
        action_regrets = []
        bet_regrets = []
        for batch_id, indices in zip(batch_ids, indices):
            self.strategy_batch_managers[player].batch_ready_events[batch_id].wait(timeout=None)
            batch_action_regrets, batch_bet_regrets = self._retrieve_strategy_batch_result(player, batch_id, indices)
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

