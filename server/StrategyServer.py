import os
import logging
import torch
import numpy as np
from rpc import RL_pb2_grpc
from threading import Lock
from rpc.RL_pb2 import Empty, IntMessage
from server.util import Reservoir, Learner
from server.networks import StrategyNetwork
from config import LINEAR_CFR, STRATEGY_LEARNING_RATE, STRATEGY_WEIGHT_DECAY, MAX_TRAIN_BATCH_SIZE, OBS_SHAPE, N_BET_BUCKETS, N_EPOCHS


class StrategyLearner(RL_pb2_grpc.StrategyLearnerServicer, Learner):
    def __init__(self, gpu_lock, ready):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logging.info('Regret learner using device %s' % self.device)
        super(StrategyLearner, self).__init__(self.device)
        self.strategy_lock = Lock()
        self.strategy_reservoir = Reservoir(save_weights=LINEAR_CFR)
        self.strategy_iterations = 0

        self.gpu_lock = gpu_lock
        self.strategy_net = StrategyNetwork(self.device).to(self.device)
        self.strategy_optimizer_fn = torch.optim.Adam
        self.strategy_scheduler_fn = torch.optim.lr_scheduler.OneCycleLR
        self.strategy_loss = torch.nn.KLDivLoss(reduction='batchmean')

        self.gpu_lock.acquire()
        self._load_initial_states()
        self.gpu_lock.release()
        ready.set()

    def _load_initial_states(self):
        states = os.listdir('states/strategy')
        i = 0
        base_name = 'strategy_net_'
        while True:
            if base_name + str(i) in states:
                i += 1
            else:
                if i == 0:
                    torch.save(self.strategy_net.state_dict(), 'states/strategy/strategy_net_0')
                else:
                    i -= 1
                break

        if i == 0:
            torch.save(self.strategy_net.state_dict(), 'states/strategy/strategy_net_0')

        self.strategy_iterations = i
        base_path = 'reservoirs/'
        success = self.strategy_reservoir.load_from_disk(base_path + 'sample_count.npy', base_path + 'obs.npy',
                                                         base_path + 'obs_count.npy', base_path + 'actions.npy', base_path + 'bets.npy')
        if success:
            logging.info("Succesfully recovered strategy reservoir")
        else:
            logging.info("Failed to recover strategy reservoir")
            self.strategy_iterations = 0

        logging.info("Starting from strategy iteration %d" % self.strategy_iterations)

    def AddStrategies(self, request, context):
        logging.debug("received %d strategies" % request.shape)
        observations = np.frombuffer(request.observations, dtype=np.float32).reshape(request.shape, request.sequence_length, OBS_SHAPE)
        obs_counts = np.frombuffer(request.observation_counts, dtype=np.int32).reshape(request.shape, 1)
        action_strategy = np.frombuffer(request.action_data, dtype=np.float32).reshape(request.shape, 4)
        bet_strategy = np.frombuffer(request.bet_data, dtype=np.float32).reshape(request.shape, N_BET_BUCKETS)
        self.strategy_lock.acquire()
        weights = None
        if LINEAR_CFR:
            weights = np.expand_dims(np.repeat(self.strategy_iterations+1, observations.shape[0]), 1)
        self.strategy_reservoir.add(observations, obs_counts, action_strategy, bet_strategy, weights=weights)
        self.strategy_lock.release()
        return Empty()

    def TrainStrategy(self, request, context):
        self.strategy_lock.acquire()
        n_samples = self.strategy_reservoir.sample_count
        logging.info("Training strategy network with %d samples" % n_samples)
        all_indices = np.arange(n_samples, dtype=np.int64)
        train_indices = np.random.choice(all_indices, int(0.8 * n_samples), replace=False)
        validation_indices = np.setdiff1d(all_indices, train_indices)
        self.gpu_lock.acquire()
        self.strategy_net.load_state_dict(torch.load('states/strategy/strategy_net_%d' % self.strategy_iterations))
        optimizer = self.strategy_optimizer_fn(self.strategy_net.parameters(), lr=STRATEGY_LEARNING_RATE, weight_decay=STRATEGY_WEIGHT_DECAY)
        scheduler = self.strategy_scheduler_fn(optimizer=optimizer, max_lr=STRATEGY_LEARNING_RATE, total_steps=None,
                                               epochs=int(N_EPOCHS), steps_per_epoch=MAX_TRAIN_BATCH_SIZE, pct_start=0.3,
                                               anneal_strategy='cos', cycle_momentum=True,
                                               base_momentum=0.85, max_momentum=0.95, div_factor=25.0,
                                               final_div_factor=1000.0, last_epoch=-1)

        obs = self.strategy_reservoir.obs
        obs_count = self.strategy_reservoir.obs_count
        actions = self.strategy_reservoir.actions
        bets = self.strategy_reservoir.bets
        iteration = self.strategy_iterations+1
        weights = None
        if LINEAR_CFR:
            weights = self.strategy_reservoir.weights
        self._training_loop(self.strategy_net, optimizer, self.strategy_loss, obs, obs_count, actions, bets, train_indices,
                            validation_indices, scheduler, 'iter', iteration, weights)
        self.strategy_iterations += 1
        torch.save(self.strategy_net.state_dict(), 'states/strategy/strategy_net_%d' % self.strategy_iterations)
        self.gpu_lock.release()
        base_path = 'reservoirs/'
        self.strategy_reservoir.save_to_disk(base_path + 'sample_count.npy', base_path + 'obs.npy', base_path + 'obs_count.npy',
                                             base_path + 'actions.npy', base_path + 'bets.npy')
        self.strategy_lock.release()
        return Empty()

    def AvailableStrategies(self, request, context):
        return IntMessage(value=self.strategy_iterations)