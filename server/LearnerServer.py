import os
import logging
import torch
import numpy as np
from rpc import RL_pb2_grpc
from threading import Lock
from rpc.RL_pb2 import Empty, IntMessage
from server.util import Reservoir, Learner
from server.networks import RegretNetwork, StrategyNetwork
from config import REGRET_LEARNING_RATE, REGRET_WEIGHT_DECAY, STRATEGY_LEARNING_RATE, STRATEGY_WEIGHT_DECAY, MAX_TRAIN_BATCH_SIZE, \
    OBS_SHAPE, N_BET_BUCKETS, N_EPOCHS, SEQUENCE_LENGTH, N_ACTIONS, N_PLAYERS, LINEAR_CFR


class RegretLearner(RL_pb2_grpc.RegretLearnerServicer, Learner):
    def __init__(self, player_list, gpu_lock, ready):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logging.info('Regret learner using device %s' % self.device)
        super(RegretLearner, self).__init__(self.device)
        self.regret_locks = {player: Lock() for player in player_list}
        self.regret_reservoirs = {player: Reservoir() for player in player_list}
        self.regret_iterations = np.zeros(N_PLAYERS)

        self.gpu_lock = gpu_lock
        self.regret_net = RegretNetwork(self.device).to(self.device)
        self.regret_optimizer_fn = torch.optim.Adam
        self.regret_scheduler_fn = torch.optim.lr_scheduler.OneCycleLR
        self.regret_loss = torch.nn.MSELoss()

        self.gpu_lock.acquire()
        self._load_initial_states(player_list)
        self.gpu_lock.release()
        ready.set()

    def _load_initial_states(self, player_list):
        try:
            self.regret_iterations = np.load('states/regret/regret_iterations.npy')
        except FileNotFoundError:
            pass
        states = os.listdir('states/regret')
        if 'regret_net_initial' not in states:
            logging.info("Pre training initial regret network to 0 regret")
            n_samples = int(1e4)
            junkobs = np.random.randint(0, 200, [n_samples, SEQUENCE_LENGTH, OBS_SHAPE])
            junkcounts = np.random.randint(1, 6, [n_samples, 1])
            a_labels = np.zeros([n_samples, N_ACTIONS])
            b_labels = np.zeros([n_samples, N_BET_BUCKETS])
            optimizer = self.regret_optimizer_fn(self.regret_net.parameters(), lr=REGRET_LEARNING_RATE, weight_decay=REGRET_WEIGHT_DECAY)
            scheduler = self.regret_scheduler_fn(optimizer=optimizer, max_lr=REGRET_LEARNING_RATE, total_steps=None,
                                                 epochs=int(N_EPOCHS), steps_per_epoch=MAX_TRAIN_BATCH_SIZE, pct_start=0.3,
                                                 anneal_strategy='cos', cycle_momentum=True,
                                                 base_momentum=0.85, max_momentum=0.95, div_factor=25.0,
                                                 final_div_factor=1000.0, last_epoch=-1)
            all_indices = np.arange(n_samples, dtype=np.int64)
            train_indices = np.random.choice(all_indices, int(0.8*n_samples), replace=False)
            validation_indices = np.setdiff1d(all_indices, train_indices)
            self._training_loop(self.regret_net, optimizer, self.regret_loss, junkobs, junkcounts, a_labels, b_labels, train_indices,
                                validation_indices, scheduler, 'iter')
            torch.save(self.regret_net.state_dict(), 'states/regret/regret_net_initial')

        for player in player_list:
            torch.save(self.regret_net.state_dict(), 'states/regret/regret_net_player_%d' % player)
            if 'player_%d' % player not in os.listdir('reservoirs'):
                os.makedirs('reservoirs/player_%d' % player)

            base_path = 'reservoirs/player_%d/' % player
            success = self.regret_reservoirs[player].load_from_disk(base_path + 'regret_samples.npy', base_path + 'regret_obs.npy',
                                                          base_path + 'regret_obs_count.npy', base_path + 'regret_actions.npy',
                                                          base_path + 'regret_bets.npy')
            if success:
                logging.info('Succesfully recovered regret reservoir for player %d' % player)
            else:
                logging.info('Failed to recover regret reservoir for player %d' % player)

    def AddRegrets(self, request, context):
        player = int(request.player)
        logging.debug("received %d regrets for player %d" % (request.shape, player))
        observations = np.frombuffer(request.observations, dtype=np.float32).reshape(request.shape, request.sequence_length, OBS_SHAPE)
        obs_counts = np.frombuffer(request.observation_counts, dtype=np.int32).reshape(request.shape, 1)
        action_regrets = np.frombuffer(request.action_data, dtype=np.float32).reshape(request.shape, 4)
        bet_regrets = np.frombuffer(request.bet_data, dtype=np.float32).reshape(request.shape, N_BET_BUCKETS)
        self.regret_locks[player].acquire()
        weights = None
        if LINEAR_CFR:
            weights = np.expand_dims(np.repeat(self.regret_iterations[player], observations.shape[0]), 1)
        self.regret_reservoirs[player].add(observations, obs_counts, action_regrets, bet_regrets, weights)
        self.regret_locks[player].release()
        return Empty()

    def TrainRegrets(self, request, context):
        player = int(request.value)
        self.regret_locks[player].acquire()
        n_samples = self.regret_reservoirs[player].sample_count
        logging.info("Training regret network for player %d with %d samples" % (player, n_samples))
        all_indices = np.arange(n_samples, dtype=np.int64)
        train_indices = np.random.choice(all_indices, int(0.8*n_samples), replace=False)
        validation_indices = np.setdiff1d(all_indices, train_indices)
        self.gpu_lock.acquire()
        self.regret_net.load_state_dict(torch.load('states/regret/regret_net_initial'))
        optimizer = self.regret_optimizer_fn(self.regret_net.parameters(), lr=REGRET_LEARNING_RATE, weight_decay=REGRET_WEIGHT_DECAY)
        scheduler = self.regret_scheduler_fn(optimizer=optimizer, max_lr=REGRET_LEARNING_RATE, total_steps=None,
                                               epochs=int(N_EPOCHS), steps_per_epoch=MAX_TRAIN_BATCH_SIZE, pct_start=0.3,
                                               anneal_strategy='cos', cycle_momentum=True,
                                               base_momentum=0.85, max_momentum=0.95, div_factor=25.0,
                                               final_div_factor=1000.0, last_epoch=-1)

        obs = self.regret_reservoirs[player].obs
        obs_count = self.regret_reservoirs[player].obs_count
        actions = self.regret_reservoirs[player].actions
        bets = self.regret_reservoirs[player].bets
        iteration = self.regret_iterations[player]
        weights = None
        if LINEAR_CFR:
            weights = self.regret_reservoirs[player].weights
        self._training_loop(self.regret_net, optimizer, self.regret_loss, obs, obs_count, actions, bets, train_indices, validation_indices,
                            scheduler, 'iter', iteration, weights)
        self.regret_iterations[player] += 1
        np.save('states/regret/regret_iterations.npy', self.regret_iterations)
        torch.save(self.regret_net.state_dict(), 'states/regret/regret_net_player_%d' % player)
        self.gpu_lock.release()
        base_path = 'reservoirs/player_%d/' % player
        self.regret_reservoirs[player].save_to_disk(
            base_path + 'regret_samples.npy',
            base_path + 'regret_obs.npy',
            base_path + 'regret_obs_count.npy',
            base_path + 'regret_actions.npy',
            base_path + 'regret_bets.npy'
        )
        self.regret_locks[player].release()
        return Empty()


class StrategyLearner(RL_pb2_grpc.StrategyLearnerServicer, Learner):
    def __init__(self, gpu_lock, ready):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logging.info('Regret learner using device %s' % self.device)
        super(StrategyLearner, self).__init__(self.device)
        self.strategy_lock = Lock()
        self.strategy_reservoir = Reservoir()
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
        self.strategy_reservoir.add(observations, obs_counts, action_strategy, bet_strategy)
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
        self._training_loop(self.strategy_net, optimizer, self.strategy_loss, obs, obs_count, actions, bets, train_indices,
                            validation_indices, scheduler, 'iter')
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