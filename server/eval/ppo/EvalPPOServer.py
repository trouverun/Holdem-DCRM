import os
import logging
import torch
import numpy as np
from rpc import RL_pb2_grpc
from queue import Queue
from server.util import BatchManager, ExperienceBuffer
from rpc.RL_pb2 import Prediction, Empty, IntMessage
from threading import Lock, Thread
from server.networks import PolicyNetwork, ValueNetwork
from config import OBS_SHAPE, PPO_BATCH_PROCESS_TIMEOUT, SEQUENCE_LENGTH, N_ACTIONS, N_BET_BUCKETS, ACTOR_LEARNING_RATE, ACTOR_WEIGHT_DECAY, \
    CRITIC_LEARNING_RATE, CRITIC_WEIGHT_DECAY, N_PPO_EPOCHS, PPO_MAX_TRAIN_BATCH_SIZE, PATIENCE, PPO_TRAINING_TIMEOUT, PPO_CLIP_VALUE, \
    PPO_RNN_HIDDENS, PPO_EXPERIENCE_BUFFER_SIZE, PPO_MAX_INFERENCE_BATCH_SIZE
from multiprocessing import Event


class ActorCritic(RL_pb2_grpc.EvalPPOServicer):
    def __init__(self, gpu_lock, ready):
        self.gpu_lock = gpu_lock
        self.player_strategy_locks = Lock()
        self.strategy_batch_que = Queue()
        self.strategy_batch_managers = BatchManager(PPO_MAX_INFERENCE_BATCH_SIZE, self.strategy_batch_que)
        self.experience_buffer_lock = Lock()
        self.experience_batch_ready = Event()
        self.data_in_experience_buffer = Event()
        self.experience_buffer = ExperienceBuffer(PPO_EXPERIENCE_BUFFER_SIZE, PPO_MAX_TRAIN_BATCH_SIZE, self.data_in_experience_buffer, self.experience_batch_ready)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logging.info('Pytorch using device %s' % self.device)
        self.actor_net = PolicyNetwork(self.device, n_hiddens=PPO_RNN_HIDDENS).to(self.device)
        self.critic_net = ValueNetwork(self.device).to(self.device)
        self.actor_optimizer_fn = torch.optim.Adam
        self.actor_optimizer = self.actor_optimizer_fn(self.critic_net.parameters(), lr=CRITIC_LEARNING_RATE, weight_decay=CRITIC_WEIGHT_DECAY)
        self.baseline_optimizer_fn = torch.optim.Adam
        self.baseline_optimizer = self.actor_optimizer_fn(self.critic_net.parameters(), lr=CRITIC_LEARNING_RATE, weight_decay=CRITIC_WEIGHT_DECAY)
        self.baseline_loss = torch.nn.MSELoss()

        self.gpu_lock.acquire()
        # self.regret_net.to(self.device)
        self._load_initial_states()
        # self.regret_net.to('cpu')
        torch.cuda.empty_cache()
        self.gpu_lock.release()
        Thread(target=self._process_strategy_batch_thread, args=()).start()
        Thread(target=self._best_response_training_thread, args=()).start()
        ready.set()

    def _load_initial_states(self):
        states = os.listdir('states/ppo')
        n_samples = int(1e4)
        junkobs = np.random.randint(0, 200, [n_samples, SEQUENCE_LENGTH, OBS_SHAPE])
        junkcounts = np.random.randint(1, 6, [n_samples, 1])

        if 'actor_net_initial' not in states:
            logging.info("Pre training initial actor network to uniform probability")
            a_labels = np.log(1/N_ACTIONS*np.ones([n_samples, N_ACTIONS]))
            b_labels = np.log(1/N_BET_BUCKETS*np.ones([n_samples, N_BET_BUCKETS]))
            policy_loss_fn = torch.nn.KLDivLoss(reduction='batchmean')
            all_indices = np.arange(n_samples, dtype=np.int64)
            train_indices = np.random.choice(all_indices, int(0.8 * n_samples), replace=False)
            validation_indices = np.setdiff1d(all_indices, train_indices)
            optimizer = self.actor_optimizer_fn(self.actor_net.parameters(), lr=ACTOR_LEARNING_RATE, weight_decay=ACTOR_WEIGHT_DECAY)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=ACTOR_LEARNING_RATE, total_steps=None,
                                                  epochs=int(N_PPO_EPOCHS), steps_per_epoch=int(len(train_indices) / PPO_MAX_TRAIN_BATCH_SIZE)+1, pct_start=0.3,
                                                  anneal_strategy='cos', cycle_momentum=True,
                                                  base_momentum=0.85, max_momentum=0.95, div_factor=25.0,
                                                  final_div_factor=1000.0, last_epoch=-1)
            self._initialization_loop(self.actor_net, optimizer, junkobs, junkcounts, a_labels, b_labels, None, train_indices, validation_indices, scheduler, policy_loss_fn)
            torch.save(self.actor_net.state_dict(), 'states/ppo/actor_net_initial')
        else:
            self.actor_net.load_state_dict(torch.load('states/ppo/actor_net_initial'))
        torch.save(self.actor_net.state_dict(), 'states/ppo/actor_net')

        if 'critic_net_initial' not in states:
            logging.info("Pre training initial critic network to 0 values")
            values = np.zeros([n_samples, 1])

            value_loss_fn = torch.nn.MSELoss()
            all_indices = np.arange(n_samples, dtype=np.int64)
            train_indices = np.random.choice(all_indices, int(0.8 * n_samples), replace=False)
            validation_indices = np.setdiff1d(all_indices, train_indices)
            optimizer = self.baseline_optimizer_fn(self.critic_net.parameters(), lr=CRITIC_LEARNING_RATE, weight_decay=CRITIC_WEIGHT_DECAY)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=CRITIC_LEARNING_RATE, total_steps=None,
                                                  epochs=int(N_PPO_EPOCHS), steps_per_epoch=int(len(train_indices) / PPO_MAX_TRAIN_BATCH_SIZE)+1, pct_start=0.3,
                                                  anneal_strategy='cos', cycle_momentum=True,
                                                  base_momentum=0.85, max_momentum=0.95, div_factor=25.0,
                                                  final_div_factor=1000.0, last_epoch=-1)
            self._initialization_loop(self.critic_net, optimizer, junkobs, junkcounts, None, None, values, train_indices, validation_indices, scheduler, value_loss_fn)
            torch.save(self.critic_net.state_dict(), 'states/ppo/critic_net_initial')
        else:
            self.critic_net.load_state_dict(torch.load('states/ppo/critic_net_initial'))
        torch.save(self.critic_net.state_dict(), 'states/ppo/critic_net')

    def _initialization_loop(self, net, optim, obs, obs_counts, actions, bets, values, train_indices, validation_indices, scheduler, loss_fn):
        if values is None:
            mode = 'actor'
        else:
            mode = 'critic'
        best = None
        no_improvement = 0
        for epoch in range(N_PPO_EPOCHS):
            running_train_loss = 0
            running_validation_loss = 0
            train_batch_start_i = 0
            valid_batch_start_i = 0
            net.train()
            while True:
                batch_end_i = train_batch_start_i + PPO_MAX_TRAIN_BATCH_SIZE
                if batch_end_i > len(train_indices):
                    batch_end_i = len(train_indices)
                batch_indices = train_indices[train_batch_start_i:batch_end_i]
                train_batch_start_i += PPO_MAX_TRAIN_BATCH_SIZE
                optim.zero_grad()
                x = torch.from_numpy(obs[batch_indices, :, :]).type(torch.FloatTensor).to(self.device)
                x_counts = torch.from_numpy(obs_counts[batch_indices]).type(torch.LongTensor).squeeze(1)
                if mode == 'actor':
                    y_action = torch.from_numpy(actions[batch_indices]).type(torch.FloatTensor).to(self.device)
                    y_bet = torch.from_numpy(bets[batch_indices]).type(torch.FloatTensor).to(self.device)
                    action_pred, bet_pred = net(x, x_counts)
                    loss_a = loss_fn(action_pred, y_action)
                    loss_b = loss_fn(bet_pred, y_bet)
                    loss = loss_a + loss_b
                else:
                    y_value = torch.from_numpy(values[batch_indices]).type(torch.FloatTensor).to(self.device)
                    value_pred = net(x, x_counts)
                    loss = loss_fn(value_pred, y_value)
                loss.backward()
                optim.step()
                running_train_loss += loss.item()
                if batch_end_i == len(train_indices):
                    break
                scheduler.step()
            net.eval()
            with torch.no_grad():
                while True:
                    batch_end_i = valid_batch_start_i + PPO_MAX_TRAIN_BATCH_SIZE
                    if batch_end_i > len(validation_indices):
                        batch_end_i = len(validation_indices)
                    batch_indices = validation_indices[valid_batch_start_i:batch_end_i]
                    valid_batch_start_i += PPO_MAX_TRAIN_BATCH_SIZE
                    x = torch.from_numpy(obs[batch_indices, :, :]).type(torch.FloatTensor).to(self.device)
                    x_counts = torch.from_numpy(obs_counts[batch_indices]).type(torch.LongTensor).squeeze(1)
                    if mode == 'actor':
                        y_action = torch.from_numpy(actions[batch_indices]).type(torch.FloatTensor).to(self.device)
                        y_bet = torch.from_numpy(bets[batch_indices]).type(torch.FloatTensor).to(self.device)
                        action_pred, bet_pred = net(x, x_counts)
                        loss_a = loss_fn(action_pred, y_action)
                        loss_b = loss_fn(bet_pred, y_bet)
                        loss = loss_a + loss_b
                    else:
                        y_value = torch.from_numpy(values[batch_indices]).type(torch.FloatTensor).to(self.device)
                        value_pred = net(x, x_counts)
                        loss = loss_fn(value_pred, y_value)
                    running_validation_loss += loss.item()
                    if batch_end_i == len(validation_indices):
                        break
            running_train_loss /= len(train_indices)
            running_validation_loss /= len(validation_indices)
            logging.info("Epoch %d: train_loss: %.20f, valid_loss: %.20f (no improvement = %d)" %
                         (epoch, running_train_loss, running_validation_loss, no_improvement))
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

    def _process_strategy_batch_thread(self):
        logging.info("Started best response inference data processing thread")
        current_batch = 0
        while True:
            self.strategy_batch_managers.data_added_events[current_batch].wait(timeout=None)
            # Once any amount of data has been added to the current batch, wait until the batch is full, or until the specified timeout
            self.strategy_batch_managers.batch_full_events[current_batch].wait(timeout=PPO_BATCH_PROCESS_TIMEOUT)
            self.player_strategy_locks.acquire()
            this_batch_size = self.strategy_batch_managers.current_batch_sizes[current_batch]
            # If we arrived here by timing out above, we need to set the current batch as consumed, so that we no longer add data to it
            if this_batch_size != PPO_MAX_INFERENCE_BATCH_SIZE:
                self.strategy_batch_managers.finalize_batch()
            self.player_strategy_locks.release()
            logging.debug("Processing strategy batch size of: %d" % this_batch_size)
            observations, counts = self.strategy_batch_que.get()
            self.gpu_lock.acquire()
            #self.strategy_net.to(self.device)
            observations = torch.from_numpy(observations[:this_batch_size]).type(torch.FloatTensor).to(self.device)
            counts = torch.from_numpy(counts[:this_batch_size]).type(torch.LongTensor)
            self.actor_net.load_state_dict(torch.load('states/ppo/actor_net'))
            with torch.no_grad():
                self.actor_net.eval()
                action_predictions, bet_predictions = self.actor_net(observations, counts)
            self.player_strategy_locks.acquire()
            self.strategy_batch_managers.add_batch_results(current_batch, action_predictions.detach().cpu().numpy(), bet_predictions.detach().cpu().numpy())
            self.player_strategy_locks.release()
            #self.strategy_net.to('cpu')
            torch.cuda.empty_cache()
            self.gpu_lock.release()
            current_batch += 1

    def _best_response_training_thread(self):
        logging.info("Started best response training thread")
        while True:
            self.data_in_experience_buffer.wait()
            self.experience_batch_ready.wait(PPO_TRAINING_TIMEOUT)
            self._train_policy()

    def _add_strategy_observations_to_batch(self, observations, counts):
        self.player_strategy_locks.acquire()
        batch_ids, indices = self.strategy_batch_managers.add_data_to_batch(observations, counts)
        self.player_strategy_locks.release()
        return batch_ids, indices

    def _retrieve_strategy_batch_result(self, batch_id, indices):
        self.player_strategy_locks.acquire()
        action_policy, bet_policy, _ = self.strategy_batch_managers.get_batch_results(batch_id, indices)
        self.player_strategy_locks.release()
        return action_policy, bet_policy

    def GetStrategies(self, request, context):
        observations = np.frombuffer(request.observations, dtype=np.float32).reshape(request.shape, request.sequence_length, OBS_SHAPE)
        obs_counts = np.frombuffer(request.observation_counts, dtype=np.int32).reshape(request.shape)
        batch_ids, indices = self._add_strategy_observations_to_batch(observations, obs_counts)
        action_regrets = []
        bet_regrets = []
        for batch_id, indices in zip(batch_ids, indices):
            self.strategy_batch_managers.batch_ready_events[batch_id].wait(timeout=None)
            batch_action_regrets, batch_bet_regrets = self._retrieve_strategy_batch_result(batch_id, indices)
            action_regrets.extend(batch_action_regrets)
            bet_regrets.extend(batch_bet_regrets)
        action_bytes = np.ndarray.tobytes(np.asarray(action_regrets))
        bet_bytes = np.ndarray.tobytes(np.asarray(bet_regrets))
        return Prediction(action_prediction=action_bytes, bet_prediction=bet_bytes)

    def AddExperience(self, request, context):
        observations = np.frombuffer(request.observations, dtype=np.float32).reshape(request.shape, request.sequence_length, OBS_SHAPE)
        obs_counts = np.frombuffer(request.observation_counts, dtype=np.int32).reshape(request.shape)
        actions = np.frombuffer(request.action_log_probs, dtype=np.float32).reshape(request.shape, 2)
        bets = np.frombuffer(request.bet_log_probs, dtype=np.float32).reshape(request.shape, 2)
        rewards = np.frombuffer(request.rewards, dtype=np.float32).reshape(request.shape)
        self.experience_buffer_lock.acquire()
        self.experience_buffer.add_experience(observations, obs_counts, actions, bets, rewards)
        self.experience_buffer_lock.release()
        return Empty()

    def _train_policy(self):
        self.experience_buffer_lock.acquire()
        observations, obs_counts, actions, bets, rewards = self.experience_buffer.get_batch()
        logging.info(rewards.mean())
        self.experience_buffer_lock.release()

        n_samples = len(observations)
        if n_samples < int(PPO_MAX_TRAIN_BATCH_SIZE / 10):
            return

        self.gpu_lock.acquire()
        observations = torch.from_numpy(observations).type(torch.FloatTensor).to(self.device)
        obs_counts = torch.from_numpy(obs_counts).type(torch.LongTensor)
        actions = torch.from_numpy(actions).type(torch.FloatTensor).to(self.device)
        bets = torch.from_numpy(bets).type(torch.FloatTensor).to(self.device)
        rewards = torch.from_numpy(rewards).type(torch.FloatTensor).to(self.device)

        self.baseline_optimizer.zero_grad()
        self.critic_net.train()
        baseline = self.critic_net(observations, obs_counts).squeeze(1)
        bl_loss = 0.5 * self.baseline_loss(baseline, rewards)
        bl_loss.backward()
        self.baseline_optimizer.step()
        bl_loss = bl_loss.detach()

        self.actor_optimizer.zero_grad()
        with torch.no_grad():
            q_values = rewards
            advantages = q_values - baseline.detach()

        old_log_pi_act = actions[:, 1]
        old_log_pi_bet = bets[:, 1]
        action_types = actions[:, 0].type(torch.LongTensor).to(self.device)
        bet_types = bets[:, 0].type(torch.LongTensor).to(self.device)
        self.actor_net.train()
        action_distributions, bet_distributions = self.actor_net(observations, obs_counts, return_dist=True)
        log_pi_act = action_distributions.log_prob(action_types)#[torch.arange(0, action_distributions.shape[0]), action_types]
        log_pi_bet = bet_distributions.log_prob(bet_types)#[torch.arange(0, bet_distributions.shape[0]), bet_types]

        r_act = log_pi_act - old_log_pi_act
        r_bet = log_pi_bet - old_log_pi_bet
        condition = action_types != 2
        r_bet = torch.where(condition.to(self.device), torch.zeros_like(r_bet), r_bet)
        r = torch.exp(r_act + r_bet)
        surr1 = r * advantages
        surr2 = torch.clamp(r, 1 - PPO_CLIP_VALUE, 1 + PPO_CLIP_VALUE) * advantages

        loss = -torch.mean(torch.min(surr1, surr2) - 0.01 * (action_distributions.entropy() + bet_distributions.entropy()))
        loss.backward()
        self.actor_optimizer.step()
        loss = loss.detach()

        logging.info("Actor loss: %.20f, Baseline loss: %.20f \t (batch size: %d)" % (loss, bl_loss, n_samples))
        torch.cuda.empty_cache()
        self.gpu_lock.release()

    def TrajectoriesLeft(self, request, context):
        self.experience_buffer_lock.acquire()
        value = self.experience_buffer.n_samples
        self.experience_buffer_lock.release()
        return IntMessage(value=value)

    def ResetBestResponse(self, request, context):
        logging.info("Resetting best response to uniform policy / 0 value baseline")
        self.gpu_lock.acquire()
        self.actor_optimizer = self.actor_optimizer_fn(self.critic_net.parameters(), lr=CRITIC_LEARNING_RATE, weight_decay=CRITIC_WEIGHT_DECAY)
        self.baseline_optimizer = self.actor_optimizer_fn(self.critic_net.parameters(), lr=CRITIC_LEARNING_RATE, weight_decay=CRITIC_WEIGHT_DECAY)
        self.actor_net.load_state_dict(torch.load('states/ppo/actor_net_initial'))
        torch.save(self.actor_net.state_dict(), 'states/ppo/actor_net')
        self.critic_net.load_state_dict(torch.load('states/ppo/critic_net_initial'))
        torch.save(self.critic_net.state_dict(), 'states/ppo/critic_net')
        self.gpu_lock.release()
        return Empty()