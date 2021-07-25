import logging
import torch
import numpy as np
from rpc import RL_pb2_grpc
from rpc.RL_pb2 import Empty
from collections import namedtuple
from queue import Queue
from threading import Lock, Thread
from config import N_PLAYERS, REGRET_LEARNING_RATE, REGRET_WEIGHT_DECAY, STRATEGY_LEARNING_RATE, STRATEGY_WEIGHT_DECAY, MAX_TRAIN_BATCH_SIZE, \
    OBS_SHAPE, RESERVOIR_SIZE, N_BET_BUCKETS, N_EPOCHS, PATIENCE, SEQUENCE_LENGTH, N_ACTIONS
from networks import RegretNetwork, StrategyNetwork

SampledData = namedtuple("SampledData", "obs count actions bets")


class Learner(RL_pb2_grpc.LearnerServicer):
    def __init__(self, gpu_lock):
        # ---------------------------------- Regret ----------------------------------
        self.regret_locks = [Lock() for _ in range(N_PLAYERS)]
        self.regret_sample_que = [Queue() for _ in range(N_PLAYERS)]
        self.regret_sample_counts = np.zeros(N_PLAYERS, dtype=np.int64)
        self.regret_observations = np.zeros([N_PLAYERS, RESERVOIR_SIZE, SEQUENCE_LENGTH, OBS_SHAPE], dtype=np.float32)
        self.regret_observation_counts = np.zeros([N_PLAYERS, RESERVOIR_SIZE, 1])
        self.regret_actions = np.zeros([N_PLAYERS, RESERVOIR_SIZE, N_ACTIONS], dtype=np.float32)
        self.regret_bets = np.zeros([N_PLAYERS, RESERVOIR_SIZE, N_BET_BUCKETS], dtype=np.float32)
        # ---------------------------------- Strategy ---------------------------------
        self.strategy_lock = Lock()
        self.strategy_sample_que = Queue()
        self.strategy_sample_count = 0
        self.strategy_observations = np.zeros([RESERVOIR_SIZE, SEQUENCE_LENGTH, OBS_SHAPE], dtype=np.float32)
        self.strategy_observation_counts = np.zeros([RESERVOIR_SIZE, 1])
        self.strategy_actions = np.zeros([RESERVOIR_SIZE, N_ACTIONS], dtype=np.float32)
        self.strategy_bets = np.zeros([RESERVOIR_SIZE, N_BET_BUCKETS], dtype=np.float32)
        # -----------------------------------------------------------------------------
        self.gpu_lock = gpu_lock
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.regret_net = RegretNetwork(self.device).to(self.device)
        self.regret_optimizer_fn = torch.optim.Adam
        self.regret_scheduler_fn = torch.optim.lr_scheduler.OneCycleLR
        self.regret_loss = torch.nn.MSELoss()
        self.strategy_net = StrategyNetwork(self.device).to(self.device)
        self.strategy_optimizer_fn = torch.optim.Adam
        self.strategy_scheduler_fn = torch.optim.lr_scheduler.OneCycleLR
        self.strategy_loss = torch.nn.KLDivLoss(reduction='batchmean')
        try:
            self.state = np.load('../states/info.npy')
        except FileNotFoundError:
            self.state = np.zeros(1)
        logging.info('Starting from strategy iteration %d' % self.state[0])
        self.gpu_lock.acquire()
        self._load_initial_states()
        self.gpu_lock.release()
        logging.info('Pytorch using device %s' % self.device)
        for i in range(N_PLAYERS):
            Thread(target=self._reservoir_sample_regrets, args=(i,)).start()
        Thread(target=self._reservoir_sample_strategy, args=()).start()

    def _load_initial_states(self):
        torch.save(self.regret_net.state_dict(), '../states/regret_net_initial')
        for player in range(N_PLAYERS):
            torch.save(self.regret_net.state_dict(), '../states/regret_net_player_%d' % player)
            try:
                self.regret_sample_counts[player] = np.load('../reservoirs/regret_samples_player_%d.npy' % player)
                self.regret_observations[player] = np.load('../reservoirs/regret_reservoir_obs_player_%d.npy' % player).reshape(RESERVOIR_SIZE, SEQUENCE_LENGTH, OBS_SHAPE)
                self.regret_observation_counts[player] = np.load('../reservoirs/regret_reservoir_obs_count_player_%d.npy' % player)
                self.regret_actions[player] = np.load('../reservoirs/regret_reservoir_act_player_%d.npy' % player)
                self.regret_bets[player] = np.load('../reservoirs/regret_reservoir_bet_player_%d.npy' % player)
            except FileNotFoundError:
                pass
        try:
            self.strategy_net.load_state_dict(torch.load('../states/strategy_net_%d' % self.state[0]))
        except FileNotFoundError:
            if self.state[0] != 0:
                logging.info("Unable to load strategy network or optimizer from memory, starting from scratch")
                self.state = np.zeros(1)
            torch.save(self.strategy_net.state_dict(), '../states/strategy_net_%d' % self.state[0])
        try:
            self.strategy_sample_count = np.load('../reservoirs/strategy_samples.npy')
            self.strategy_observations = np.load('../reservoirs/strategy_reservoir_obs.npy').reshape(RESERVOIR_SIZE, SEQUENCE_LENGTH, OBS_SHAPE)
            self.strategy_observation_counts = np.load('../reservoirs/strategy_reservoir_obs_count.npy')
            self.strategy_actions = np.load('../reservoirs/strategy_reservoir_act.npy')
            self.strategy_bets = np.load('../reservoirs/strategy_reservoir_bet.npy')
        except FileNotFoundError:
            pass

    def _reservoir_sample_regrets(self, player):
        logging.info("Started regret reservoir sampling thread for player %d", player)
        while True:
            item = self.regret_sample_que[player].get()
            self.regret_locks[player].acquire()
            count = self.regret_sample_counts[player]
            if count < RESERVOIR_SIZE:
                self.regret_observations[player][count] = item.obs
                self.regret_observation_counts[player][count] = item.count
                self.regret_actions[player][count] = item.actions
                self.regret_bets[player][count] = item.bets
                self.regret_sample_counts[player] += 1
            elif np.random.uniform() > (1 - RESERVOIR_SIZE / self.regret_sample_counts[player]):
                to_replace = np.random.randint(0, RESERVOIR_SIZE)
                self.regret_observations[player][to_replace] = item.obs
                self.regret_observation_counts[player][to_replace] = item.count
                self.regret_actions[player][to_replace] = item.actions
                self.regret_bets[player][to_replace] = item.bets
            self.regret_locks[player].release()

    def AddRegrets(self, request, context):
        player = int(request.player)
        logging.debug("received %d regrets for player %d" % (request.shape, player))
        observations = np.frombuffer(request.observations, dtype=np.float32).reshape(request.shape, request.sequence_length, OBS_SHAPE)
        obs_counts = np.frombuffer(request.observation_counts, dtype=np.int32).reshape(request.shape, 1)
        action_regrets = np.frombuffer(request.action_data, dtype=np.float32).reshape(request.shape, 4)
        bet_regrets = np.frombuffer(request.bet_data, dtype=np.float32).reshape(request.shape, N_BET_BUCKETS)
        for i in range(request.shape):
            self.regret_sample_que[player].put(SampledData(observations[i], obs_counts[i], action_regrets[i], bet_regrets[i]))
        return Empty()

    def TrainRegrets(self, request, context):
        player = int(request.player)
        self.regret_locks[player].acquire()
        n_samples = self.regret_sample_counts[player]
        logging.info("Training regret network for player %d with %d samples" % (player, n_samples))
        all_indices = np.arange(n_samples, dtype=np.int64)
        train_indices = np.random.choice(all_indices, int(0.8*n_samples), replace=False)
        validation_indices = np.setdiff1d(all_indices, train_indices)
        self.gpu_lock.acquire()
        self.regret_net.load_state_dict(torch.load('../states/regret_net_initial'))
        optimizer = self.regret_optimizer_fn(self.regret_net.parameters(), lr=REGRET_LEARNING_RATE, weight_decay=REGRET_WEIGHT_DECAY)
        scheduler = self.regret_scheduler_fn(optimizer=optimizer, max_lr=REGRET_LEARNING_RATE, total_steps=None,
                                               epochs=int(N_EPOCHS), steps_per_epoch=MAX_TRAIN_BATCH_SIZE, pct_start=0.3,
                                               anneal_strategy='cos', cycle_momentum=True,
                                               base_momentum=0.85, max_momentum=0.95, div_factor=25.0,
                                               final_div_factor=1000.0, last_epoch=-1)
        self._training_loop(self.regret_net, optimizer, self.regret_loss, self.regret_observations[player],
                            self.regret_observation_counts[player], self.regret_actions[player], self.regret_bets[player], train_indices, validation_indices, scheduler, 'iter')
        torch.save(self.regret_net.state_dict(), '../states/regret_net_player_%d' % player)
        self.gpu_lock.release()
        np.save('../reservoirs/regret_samples_player_%d.npy' % player, self.regret_sample_counts)
        np.save('../reservoirs/regret_reservoir_obs_player_%d.npy' % player, self.regret_observations[player])
        np.save('../reservoirs/regret_reservoir_obs_count_player_%d.npy' % player, self.regret_observation_counts[player])
        np.save('../reservoirs/regret_reservoir_act_player_%d.npy' % player, self.regret_actions[player])
        np.save('../reservoirs/regret_reservoir_bet_player_%d.npy' % player, self.regret_bets[player])
        self.regret_locks[player].release()
        return Empty()

    def _reservoir_sample_strategy(self):
        logging.info("Started strategy reservoir sampling thread")
        while True:
            item = self.strategy_sample_que.get()
            self.strategy_lock.acquire()
            count = self.strategy_sample_count
            if count < RESERVOIR_SIZE:
                self.strategy_observations[count] = item.obs
                self.strategy_observation_counts[count] = item.count
                self.strategy_actions[count] = item.actions
                self.strategy_bets[count] = item.bets
                self.strategy_sample_count += 1
            elif np.random.uniform() > (1 - RESERVOIR_SIZE / self.strategy_sample_count):
                to_replace = np.random.randint(0, RESERVOIR_SIZE)
                self.strategy_observations[to_replace] = item.obs
                self.strategy_observation_counts[to_replace] = item.count
                self.strategy_actions[to_replace] = item.actions
                self.strategy_bets[to_replace] = item.bets
            self.strategy_lock.release()

    def AddStrategies(self, request, context):
        logging.debug("received %d strategies" % request.shape)
        observations = np.frombuffer(request.observations, dtype=np.float32).reshape(request.shape, request.sequence_length, OBS_SHAPE)
        obs_counts = np.frombuffer(request.observation_counts, dtype=np.int32).reshape(request.shape, 1)
        action_strategy = np.frombuffer(request.action_data, dtype=np.float32).reshape(request.shape, 4)
        bet_strategy = np.frombuffer(request.bet_data, dtype=np.float32).reshape(request.shape, N_BET_BUCKETS)
        for i in range(request.shape):
            self.strategy_sample_que.put(SampledData(observations[i], obs_counts[i], action_strategy[i], bet_strategy[i]))
        return Empty()

    def TrainStrategy(self, request, context):
        self.strategy_lock.acquire()
        n_samples = self.strategy_sample_count
        logging.info("Training strategy network with %d samples" % n_samples)
        all_indices = np.arange(n_samples, dtype=np.int64)
        train_indices = np.random.choice(all_indices, int(0.8 * n_samples), replace=False)
        validation_indices = np.setdiff1d(all_indices, train_indices)
        self.gpu_lock.acquire()
        self.strategy_net.load_state_dict(torch.load('../states/strategy_net_%d' % self.state[0]))
        optimizer = self.strategy_optimizer_fn(self.strategy_net.parameters(), lr=STRATEGY_LEARNING_RATE, weight_decay=STRATEGY_WEIGHT_DECAY)
        scheduler = self.strategy_scheduler_fn(optimizer=optimizer, max_lr=STRATEGY_LEARNING_RATE, total_steps=None,
                                               epochs=int(N_EPOCHS), steps_per_epoch=MAX_TRAIN_BATCH_SIZE, pct_start=0.3,
                                               anneal_strategy='cos', cycle_momentum=True,
                                               base_momentum=0.85, max_momentum=0.95, div_factor=25.0,
                                               final_div_factor=1000.0, last_epoch=-1)
        self._training_loop(self.strategy_net, optimizer, self.strategy_loss, self.strategy_observations,
                            self.strategy_observation_counts, self.strategy_actions, self.strategy_bets, train_indices, validation_indices, scheduler, 'iter')
        self.state[0] += 1
        torch.save(self.strategy_net.state_dict(), '../states/strategy_net_%d' % self.state[0])
        self.gpu_lock.release()
        np.save('../reservoirs/strategy_samples.npy', self.strategy_sample_count)
        np.save('../reservoirs/strategy_reservoir_obs.npy', self.strategy_observations)
        np.save('../reservoirs/strategy_reservoir_obs_count.npy', self.strategy_observation_counts)
        np.save('../reservoirs/strategy_reservoir_act.npy', self.strategy_actions)
        np.save('../reservoirs/strategy_reservoir_bet.npy', self.strategy_bets)
        np.save('../states/info.npy', self.state)
        self.strategy_lock.release()
        return Empty()

    def _training_loop(self, net, optim, loss_fn, obs, obs_counts, actions, bets, train_indices, validation_indices, scheduler, step_point):
        if len(train_indices) == 0 or len(validation_indices) == 0:
            raise Exception("Received empty training tensors, this means there are no sampled regrets or strategies. "
                            "Reduce the CLIENT_SAMPLES_BATCH_SIZE value or increase the effective traversals per iteration (k value).")
        best = None
        no_improvement = 0
        for epoch in range(N_EPOCHS):
            running_train_loss = 0
            running_validation_loss = 0
            train_batch_start_i = 0
            valid_batch_start_i = 0
            net.train()
            while True:
                batch_end_i = train_batch_start_i + MAX_TRAIN_BATCH_SIZE
                if batch_end_i > len(train_indices):
                    batch_end_i = len(train_indices)
                batch_indices = train_indices[train_batch_start_i:batch_end_i]
                train_batch_start_i += MAX_TRAIN_BATCH_SIZE
                x = torch.from_numpy(obs[batch_indices, :, :]).type(torch.FloatTensor).to(self.device)
                x_counts = torch.from_numpy(obs_counts[batch_indices]).type(torch.LongTensor).squeeze(1)
                y_action = torch.from_numpy(actions[batch_indices]).type(torch.FloatTensor).to(self.device)
                y_bet = torch.from_numpy(bets[batch_indices]).type(torch.FloatTensor).to(self.device)
                optim.zero_grad()
                action_pred, bet_pred = net(x, x_counts)
                loss_a = loss_fn(action_pred, y_action)
                loss_b = loss_fn(bet_pred, y_bet)
                loss = loss_a + loss_b
                loss.backward()
                optim.step()
                running_train_loss += loss.item()
                if batch_end_i == len(train_indices):
                    break
                if step_point == 'iter':
                    scheduler.step()
            net.eval()
            with torch.no_grad():
                while True:
                    batch_end_i = valid_batch_start_i + MAX_TRAIN_BATCH_SIZE
                    if batch_end_i > len(validation_indices):
                        batch_end_i = len(validation_indices)
                    batch_indices = validation_indices[valid_batch_start_i:batch_end_i]
                    valid_batch_start_i += MAX_TRAIN_BATCH_SIZE
                    x = torch.from_numpy(obs[batch_indices, :, :]).type(torch.FloatTensor).to(self.device)
                    x_counts = torch.from_numpy(obs_counts[batch_indices]).type(torch.LongTensor).squeeze(1)
                    y_action = torch.from_numpy(actions[batch_indices]).type(torch.FloatTensor).to(self.device)
                    y_bet = torch.from_numpy(bets[batch_indices]).type(torch.FloatTensor).to(self.device)
                    action_pred, bet_pred = net(x, x_counts)
                    loss_a = loss_fn(action_pred, y_action)
                    loss_b = loss_fn(bet_pred, y_bet)
                    loss = loss_a + loss_b
                    running_validation_loss += loss.item()
                    if batch_end_i == len(validation_indices):
                        break
            running_train_loss /= len(train_indices)
            running_validation_loss /= len(validation_indices)
            logging.info("Epoch %d: train_loss: %.20f, valid_loss: %.20f" % (epoch, running_train_loss, running_validation_loss))
            if best is None:
                best = running_validation_loss
            else:
                if running_validation_loss < best:
                    best = running_validation_loss
                    no_improvement = 0
                else:
                    no_improvement += 1
            if no_improvement > PATIENCE:
                break
            if step_point == 'epoch':
                scheduler.step()
