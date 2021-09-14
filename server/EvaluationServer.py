import os
import logging
import torch
import numpy as np
from rpc import RL_pb2_grpc
from threading import Lock, Thread
from queue import Queue
from server.util import BatchManager
from rpc.RL_pb2 import Empty, EvalPrediction
from server.util import Reservoir
from server.networks import EvaluatorNetwork
from config import MAX_TRAIN_BATCH_SIZE, OBS_SHAPE, N_BET_BUCKETS, N_VALUE_P_EPOCHS, PATIENCE, SEQUENCE_LENGTH, N_ACTIONS, INITIAL_VALUE, VALUE_P_LEARNING_RATE, VALUE_P_WEIGHT_DECAY, DATA_PROCESS_TIMEOUT, MAX_INFERENCE_BATCH_SIZE


class EvaluationServer(RL_pb2_grpc.EvaluatorServicer):
    def __init__(self, gpu_lock, ready):
        self.gpu_lock = gpu_lock
        self.value_p_batch_que = Queue()
        self.value_p_batch_managers = BatchManager(self.value_p_batch_que, True)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logging.info('Pytorch using device %s' % self.device)
        self.value_p_net = EvaluatorNetwork(self.device).to(self.device)
        self.value_p_optimizer_fn = torch.optim.Adam
        self.value_p_scheduler_fn = torch.optim.lr_scheduler.OneCycleLR
        self.value_loss_fn = torch.nn.MSELoss()
        self.prior_loss_fn = torch.nn.KLDivLoss(reduction='batchmean')
        self.value_p_lock = Lock()
        self.value_p_reservoir = Reservoir(save_value=True)
        self.value_p_iterations = np.zeros(1)
        Thread(target=self._process_value_p_batch_thread, args=()).start()

        self.gpu_lock.acquire()
        self._load_initial_states()
        self.gpu_lock.release()
        ready.set()

    def _load_initial_states(self):
        states = os.listdir('states/value_p')
        if 'value_p_net_initial' not in states:
            logging.info("Pre training initial value_p network to 0 value and uniform p")
            n_samples = int(1e4)
            junkobs = np.random.randint(0, 200, [n_samples, SEQUENCE_LENGTH, OBS_SHAPE])
            junkcounts = np.random.randint(1, 6, [n_samples, 1])
            value_labels = INITIAL_VALUE*np.ones([n_samples, 1])
            a_labels = np.log(1/N_ACTIONS*np.ones([n_samples, N_ACTIONS]))
            b_labels = np.log(1/N_BET_BUCKETS*np.ones([n_samples, N_BET_BUCKETS]))
            optimizer = self.value_p_optimizer_fn(self.value_p_net.parameters(), lr=VALUE_P_LEARNING_RATE, weight_decay=VALUE_P_WEIGHT_DECAY)
            scheduler = self.value_p_scheduler_fn(optimizer=optimizer, max_lr=VALUE_P_LEARNING_RATE, total_steps=None,
                                                 epochs=int(N_VALUE_P_EPOCHS), steps_per_epoch=MAX_TRAIN_BATCH_SIZE, pct_start=0.3,
                                                 anneal_strategy='cos', cycle_momentum=True,
                                                 base_momentum=0.85, max_momentum=0.95, div_factor=25.0,
                                                 final_div_factor=1000.0, last_epoch=-1)
            all_indices = np.arange(n_samples, dtype=np.int64)
            train_indices = np.random.choice(all_indices, int(0.8 * n_samples), replace=False)
            validation_indices = np.setdiff1d(all_indices, train_indices)
            self._training_loop(self.value_p_net, optimizer, junkobs, junkcounts, value_labels, a_labels, b_labels, train_indices,
                                validation_indices, scheduler, 'iter')
            torch.save(self.value_p_net.state_dict(), 'states/value_p/value_p_net_initial')

        torch.save(self.value_p_net.state_dict(), 'states/value_p/value_p_net')
        if 'value_p' not in os.listdir('reservoirs'):
            os.makedirs('reservoirs/value_p')
        base_path = 'reservoirs/value_p/'
        success = self.value_p_reservoir.load_from_disk(base_path + 'value_p_samples.npy', base_path + 'value_p_obs.npy',
                                                                base_path + 'value_p_obs_count.npy', base_path + 'value_pt_actions.npy',
                                                                base_path + 'value_p_bets.npy', value_loc=base_path + 'value_p_values.npy')
        if success:
            logging.info('Succesfully recovered value_p reservoir for player %d')
        else:
            logging.info('Failed to recover value_p reservoir for player %d')

    def _process_value_p_batch_thread(self):
        logging.info("Started value_p inference data processing thread")
        current_batch = 0
        while True:
            self.value_p_batch_managers.data_added_events[current_batch].wait(timeout=None)
            # Once any amount of data has been added to the current batch, wait until the batch is full, or until the specified timeout
            self.value_p_batch_managers.batch_full_events[current_batch].wait(timeout=DATA_PROCESS_TIMEOUT)
            self.value_p_lock.acquire()
            this_batch_size = self.value_p_batch_managers.current_batch_sizes[current_batch]
            # If we arrived here by timing out above, we need to set the current batch as consumed, so that we no longer add data to it
            if this_batch_size != MAX_INFERENCE_BATCH_SIZE:
                self.value_p_batch_managers.finalize_batch()
            self.value_p_lock.release()
            logging.debug("Processing value_p batch size of: %d" % this_batch_size)
            observations, counts = self.value_p_batch_que.get()
            self.gpu_lock.acquire()
            observations = torch.from_numpy(observations[:this_batch_size]).type(torch.FloatTensor).to(self.device)
            counts = torch.from_numpy(counts[:this_batch_size]).type(torch.LongTensor)
            self.value_p_net.load_state_dict(torch.load('states/value_p/value_p_net'))
            action_predictions, bet_predictions, value_predictions = self.value_p_net(observations, counts)
            self.gpu_lock.release()
            self.value_p_lock.acquire()
            self.value_p_batch_managers.add_batch_results(current_batch, action_predictions.detach().cpu().numpy(), bet_predictions.detach().cpu().numpy(), value_predictions.detach().cpu().numpy())
            self.value_p_lock.release()
            current_batch += 1

    def _add_value_p_observations_to_batch(self, player, observations, counts):
        self.value_p_lock.acquire()
        batch_ids, indices = self.value_p_batch_managers.add_data_to_batch(observations, counts)
        self.value_p_lock.release()
        return batch_ids, indices

    def _retrieve_value_p_batch_result(self, player, batch_id, indices):
        self.value_p_lock.acquire()
        action_prior, bet_prior, values = self.value_p_batch_managers.get_batch_results(batch_id, indices)
        self.value_p_lock.release()
        return action_prior, bet_prior, values

    def GetValues(self, request, context):
        player = int(request.player)
        observations = np.frombuffer(request.observations, dtype=np.float32).reshape(request.shape, request.sequence_length, OBS_SHAPE)
        obs_counts = np.frombuffer(request.observation_counts, dtype=np.int32).reshape(request.shape)
        batch_ids, indices = self._add_value_p_observations_to_batch(player, observations, obs_counts)
        action_priors = []
        bet_priors = []
        pred_values = []
        for batch_id, indices in zip(batch_ids, indices):
            self.value_p_batch_managers.batch_ready_events[batch_id].wait(timeout=None)
            action_prior, bet_prior, values = self._retrieve_value_p_batch_result(player, batch_id, indices)
            action_priors.extend(action_prior)
            bet_priors.extend(bet_prior)
            pred_values.extend(values)
        action_bytes = np.ndarray.tobytes(np.asarray(action_priors))
        bet_bytes = np.ndarray.tobytes(np.asarray(bet_priors))
        value_bytes = np.ndarray.tobytes(np.asarray(pred_values))
        return EvalPrediction(value_prediction=value_bytes, action_prior_prediction=action_bytes, bet_prior_prediction=bet_bytes)

    def AddValues(self, request, context):
        player = int(request.player)
        logging.debug("received %d value_p samples for player %d" % (request.shape, player))
        observations = np.frombuffer(request.observations, dtype=np.float32).reshape(request.shape, request.sequence_length, OBS_SHAPE)
        obs_counts = np.frombuffer(request.observation_counts, dtype=np.int32).reshape(request.shape, 1)
        action_prior = np.frombuffer(request.action_prior, dtype=np.float32).reshape(request.shape, N_ACTIONS)
        bet_prior = np.frombuffer(request.bet_prior, dtype=np.float32).reshape(request.shape, N_BET_BUCKETS)
        values = np.frombuffer(request.values, dtype=np.float32).reshape(request.shape, 1)
        self.value_p_lock.acquire()
        self.value_p_reservoir.add(observations, obs_counts, action_prior, bet_prior, values=values)
        self.value_p_lock.release()
        return Empty()

    def TrainValues(self, request, context):
        self.value_p_lock.acquire()
        n_samples = self.value_p_reservoir.sample_count
        logging.info("Training value_p network with %d samples" % (n_samples))
        all_indices = np.arange(n_samples, dtype=np.int64)
        train_indices = np.random.choice(all_indices, int(0.8 * n_samples), replace=False)
        validation_indices = np.setdiff1d(all_indices, train_indices)
        self.gpu_lock.acquire()
        self.value_p_net.load_state_dict(torch.load('states/value_p/value_p_net_initial'))
        optimizer = self.value_p_optimizer_fn(self.value_p_net.parameters(), lr=VALUE_P_LEARNING_RATE, weight_decay=VALUE_P_WEIGHT_DECAY)
        scheduler = self.value_p_scheduler_fn(optimizer=optimizer, max_lr=VALUE_P_LEARNING_RATE, total_steps=None,
                                             epochs=int(N_VALUE_P_EPOCHS), steps_per_epoch=MAX_TRAIN_BATCH_SIZE, pct_start=0.3,
                                             anneal_strategy='cos', cycle_momentum=True,
                                             base_momentum=0.85, max_momentum=0.95, div_factor=25.0,
                                             final_div_factor=1000.0, last_epoch=-1)

        obs = self.value_p_reservoir.obs
        obs_count = self.value_p_reservoir.obs_count
        values = self.value_p_reservoir.values
        actions = self.value_p_reservoir.actions
        bets = self.value_p_reservoir.bets
        self._training_loop(self.value_p_net, optimizer, obs, obs_count, values, actions, bets, train_indices, validation_indices,
                            scheduler, 'iter')
        torch.save(self.value_p_net.state_dict(), 'states/value_p/value_p_net')
        self.gpu_lock.release()
        base_path = 'reservoirs/value_p/'
        self.value_p_reservoir.save_to_disk(
            base_path + 'value_p_samples.npy',
            base_path + 'value_p_obs.npy',
            base_path + 'value_p_obs_count.npy',
            base_path + 'value_p_actions.npy',
            base_path + 'value_p_bets.npy',
            value_loc=base_path + 'value_p_values.npy'
        )
        self.value_p_lock.release()
        return Empty()

    def _training_loop(self, net, optim, obs, obs_counts, values, actions, bets, train_indices, validation_indices, scheduler, step_point, iteration=None):
        if len(train_indices) == 0 or len(validation_indices) == 0:
            raise Exception("Received empty training tensors, this means there are no sampled regrets or strategies. "
                            "Reduce the CLIENT_SAMPLES_BATCH_SIZE value or increase the effective traversals per iteration (k value).")
        best = None
        no_improvement = 0
        for epoch in range(N_VALUE_P_EPOCHS):
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
                y_value = torch.from_numpy(values[batch_indices]).type(torch.FloatTensor).to(self.device)
                y_action = torch.from_numpy(actions[batch_indices]).type(torch.FloatTensor).to(self.device)
                y_bet = torch.from_numpy(bets[batch_indices]).type(torch.FloatTensor).to(self.device)
                optim.zero_grad()
                action_pred, bet_pred, value_pred = net(x, x_counts)
                value_loss = self.value_loss_fn(value_pred, y_value)
                loss_a = self.prior_loss_fn(action_pred, y_action)
                loss_b = self.prior_loss_fn(bet_pred, y_bet)
                loss = loss_a + loss_b + value_loss
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
                    y_value = torch.from_numpy(values[batch_indices]).type(torch.FloatTensor).to(self.device)
                    y_action = torch.from_numpy(actions[batch_indices]).type(torch.FloatTensor).to(self.device)
                    y_bet = torch.from_numpy(bets[batch_indices]).type(torch.FloatTensor).to(self.device)
                    action_pred, bet_pred, value_pred = net(x, x_counts)
                    value_loss = self.value_loss_fn(value_pred, y_value)
                    loss_a = self.prior_loss_fn(action_pred, y_action)
                    loss_b = self.prior_loss_fn(bet_pred, y_bet)
                    loss = loss_a + loss_b + value_loss
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
            if step_point == 'epoch':
                scheduler.step()