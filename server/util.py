import numpy as np
import torch
import logging
from config import OBS_SHAPE, N_BET_BUCKETS, PATIENCE, SEQUENCE_LENGTH, N_ACTIONS
from multiprocessing import Event


class Learner:
    def __init__(self, device):
        self.device = device

    def _training_loop(self, epochs, batch_size, net, optim, loss_fn, obs, obs_counts, actions, bets, train_indices, validation_indices, scheduler, step_point,
                       iteration=None, weights=None):
        if len(train_indices) == 0 or len(validation_indices) == 0:
            raise Exception("Received empty training tensors, this means there are no sampled regrets or strategies. "
                            "Reduce the CLIENT_SAMPLES_BATCH_SIZE value or increase the effective traversals per iteration (k value).")
        best = None
        no_improvement = 0
        for epoch in range(epochs):
            running_train_loss = 0
            running_validation_loss = 0
            train_batch_start_i = 0
            valid_batch_start_i = 0
            net.train()
            while True:
                batch_end_i = train_batch_start_i + batch_size
                if batch_end_i > len(train_indices):
                    batch_end_i = len(train_indices)
                batch_indices = train_indices[train_batch_start_i:batch_end_i]
                train_batch_start_i += batch_size
                x = torch.from_numpy(obs[batch_indices, :, :]).type(torch.FloatTensor).to(self.device)
                x_counts = torch.from_numpy(obs_counts[batch_indices]).type(torch.LongTensor).squeeze(1)
                y_action = torch.from_numpy(actions[batch_indices]).type(torch.FloatTensor).to(self.device)
                y_bet = torch.from_numpy(bets[batch_indices]).type(torch.FloatTensor).to(self.device)
                # Discounting in the case of linear CFR
                if weights is not None:
                    batch_weights = torch.from_numpy(weights[batch_indices]).type(torch.LongTensor).to(self.device)
                    y_action = 2 / iteration * batch_weights * y_action
                    y_bet = 2 / iteration * batch_weights * y_bet
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
                    batch_end_i = valid_batch_start_i + batch_size
                    if batch_end_i > len(validation_indices):
                        batch_end_i = len(validation_indices)
                    batch_indices = validation_indices[valid_batch_start_i:batch_end_i]
                    valid_batch_start_i += batch_size
                    x = torch.from_numpy(obs[batch_indices, :, :]).type(torch.FloatTensor).to(self.device)
                    x_counts = torch.from_numpy(obs_counts[batch_indices]).type(torch.LongTensor).squeeze(1)
                    y_action = torch.from_numpy(actions[batch_indices]).type(torch.FloatTensor).to(self.device)
                    y_bet = torch.from_numpy(bets[batch_indices]).type(torch.FloatTensor).to(self.device)
                    # Discounting in the case of linear CFR
                    if weights is not None:
                        batch_weights = torch.from_numpy(weights[batch_indices]).type(torch.LongTensor).to(self.device)
                        y_action = 2 / iteration * batch_weights * y_action
                        y_bet = 2 / iteration * batch_weights * y_bet
                    action_pred, bet_pred = net(x, x_counts)
                    loss_a = loss_fn(action_pred, y_action)
                    loss_b = loss_fn(bet_pred, y_bet)
                    loss = loss_a + loss_b
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


class BatchManager:
    def __init__(self, batch_size, sample_que, predict_value=False):
        self.batch_size = batch_size
        self.sample_que = sample_que
        self.current_batch_id = 0
        self.current_batch_sizes = [0]
        self.tmp_batch_obs = np.zeros([batch_size, SEQUENCE_LENGTH, OBS_SHAPE])
        self.tmp_batch_obs_count = np.zeros([batch_size])
        self.data_added_events = {0: Event()}
        self.batch_full_events = {0: Event()}
        self.batch_ready_events = {0: Event()}
        self.batch_read_counts = []
        self.action_predictions = dict()
        self.bet_predictions = dict()
        self.predict_value = predict_value
        if predict_value:
            self.value_predictions = dict()

    def finalize_batch(self):
        current_batch = self.current_batch_id
        current_batch_size = self.current_batch_sizes[current_batch]
        self.sample_que.put((self.tmp_batch_obs[:current_batch_size], self.tmp_batch_obs_count[:current_batch_size]))
        self.tmp_batch_obs = np.zeros([self.batch_size, SEQUENCE_LENGTH, OBS_SHAPE])
        self.tmp_batch_obs_count = np.zeros([self.batch_size])
        self.current_batch_id += 1
        # Add required entries for the next batch
        self.current_batch_sizes.append(0)
        self.batch_read_counts.append(0)
        self.data_added_events[self.current_batch_id] = Event()
        self.batch_full_events[self.current_batch_id] = Event()
        self.batch_ready_events[self.current_batch_id] = Event()

    def add_data_to_batch(self, observations, counts):
        batch_ids = []
        indices = []
        current_batch = self.current_batch_id
        if len(observations) + self.current_batch_sizes[current_batch] >= self.batch_size:
            consumed = 0
            remaining = len(observations)
            current_batch_size = self.current_batch_sizes[current_batch]
            space_left = self.batch_size - current_batch_size
            while remaining >= space_left:
                batch_ids.append(self.current_batch_id)
                indices.append([*range(self.current_batch_sizes[current_batch], self.batch_size)])
                self.current_batch_sizes[current_batch] = self.batch_size
                self.tmp_batch_obs[current_batch_size:self.batch_size] = observations[consumed:consumed + space_left]

                self.tmp_batch_obs_count[current_batch_size:self.batch_size] = counts[consumed:consumed + space_left]

                self.finalize_batch()
                self.data_added_events[current_batch].set()
                self.batch_full_events[current_batch].set()
                current_batch_size = 0
                current_batch += 1
                consumed += space_left
                space_left = self.batch_size
                remaining = len(observations) - consumed
            if remaining > 0:
                batch_ids.append(self.current_batch_id)
                indices.append([*range(0, remaining)])
                self.current_batch_sizes[current_batch] = remaining
                self.tmp_batch_obs[0:remaining] = observations[consumed:consumed + remaining]
                self.tmp_batch_obs_count[0:remaining] = counts[consumed:consumed + remaining]
                self.data_added_events[current_batch].set()
        else:
            current_batch = self.current_batch_id
            batch_ids.append(current_batch)
            current_batch_size = self.current_batch_sizes[current_batch]
            indices.append([*range(current_batch_size, current_batch_size + len(observations))])
            self.tmp_batch_obs[current_batch_size:current_batch_size + len(observations)] = observations
            self.tmp_batch_obs_count[current_batch_size:current_batch_size + len(observations)] = counts
            self.current_batch_sizes[current_batch] += len(observations)
            self.data_added_events[current_batch].set()
        return batch_ids, indices

    def add_batch_results(self, batch_id, action_preds, bet_preds, value_preds=None):
        self.action_predictions[batch_id] = action_preds
        self.bet_predictions[batch_id] = bet_preds
        if value_preds is not None:
            self.value_predictions[batch_id] = value_preds
        self.batch_ready_events[batch_id].set()

    def get_batch_results(self, batch_id, indices):
        self.batch_read_counts[batch_id] += len(indices)
        action_preds = self.action_predictions[batch_id][indices]
        bet_preds = self.bet_predictions[batch_id][indices]
        value_preds = None
        if self.predict_value:
            value_preds = self.value_predictions[batch_id][indices].copy()
        if self.batch_read_counts[batch_id] == self.current_batch_sizes[batch_id]:
            self.action_predictions.pop(batch_id)
            self.bet_predictions.pop(batch_id)
            if self.predict_value:
                self.value_predictions.pop(batch_id)
            self.data_added_events.pop(batch_id)
            self.batch_full_events.pop(batch_id)
            self.batch_ready_events.pop(batch_id)
        return action_preds, bet_preds, value_preds


class ExperienceBuffer:
    def __init__(self, buffer_size, batch_size, data_in_buffer, batch_ready):
        self.n_samples = 0
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.obs = np.zeros([buffer_size, SEQUENCE_LENGTH, OBS_SHAPE], dtype=np.float32)
        self.obs_count = np.zeros([buffer_size], dtype=np.int32)
        self.actions = np.zeros([buffer_size, 2], dtype=np.float32)
        self.bets = np.zeros([buffer_size, 2], dtype=np.float32)
        self.rewards = np.zeros([buffer_size], dtype=np.float32)
        self.data_in_buffer = data_in_buffer
        self.batch_ready = batch_ready

    def add_experience(self, obs, obs_count, actions, bets, rewards):
        new_samples = len(obs)
        self.obs[self.n_samples:self.n_samples + new_samples] = obs
        self.obs_count[self.n_samples:self.n_samples + new_samples] = obs_count
        self.actions[self.n_samples:self.n_samples + new_samples] = actions
        self.bets[self.n_samples:self.n_samples + new_samples] = bets
        self.rewards[self.n_samples:self.n_samples + new_samples] = rewards
        self.n_samples += new_samples
        self.data_in_buffer.set()
        if self.n_samples >= self.batch_size:
            self.batch_ready.set()
        return True

    def get_batch(self):
        real_batch_size = min(self.n_samples, self.batch_size)
        self.n_samples -= real_batch_size
        obs = self.obs[self.n_samples:self.n_samples + real_batch_size]
        obs_count = self.obs_count[self.n_samples:self.n_samples + real_batch_size]
        actions = self.actions[self.n_samples:self.n_samples + real_batch_size]
        bets = self.bets[self.n_samples:self.n_samples + real_batch_size]
        rewards = self.rewards[self.n_samples:self.n_samples + real_batch_size]
        if self.n_samples < self.batch_size:
            if self.n_samples == 0:
                self.data_in_buffer.clear()
            self.batch_ready.clear()
        return obs, obs_count, actions, bets, rewards


class Reservoir:
    def __init__(self, size, save_weights=False, save_value=False, new_data_event=None, new_batch_event=None, batch_size=None):
        self.size = size
        self.sample_count = np.zeros(1, dtype=np.int64)
        self.obs = np.zeros([size, SEQUENCE_LENGTH, OBS_SHAPE], dtype=np.float32)
        self.obs_count = np.zeros([size, 1])
        if save_value:
            self.values = np.zeros([size, 1], dtype=np.float32)
        self.actions = np.zeros([size, N_ACTIONS], dtype=np.float32)
        self.bets = np.zeros([size, N_BET_BUCKETS], dtype=np.float32)
        if save_weights:
            self.weights = np.zeros([size, 1])
        self.new_data_event = new_data_event
        self.new_batch_event = new_batch_event
        if new_data_event is not None:
            assert batch_size is not None
            self.total_fresh_samples = 0
            self.batch_size = batch_size

    def load_from_disk(self, sample_count_loc, obs_loc, obs_count_loc, action_loc, bet_loc, weight_loc=None, value_loc=None):
        try:
            self.sample_count = np.load(sample_count_loc)
            self.obs = np.load(obs_loc).reshape(self.size, SEQUENCE_LENGTH, OBS_SHAPE)
            self.obs_count = np.load(obs_count_loc)
            self.actions = np.load(action_loc)
            self.bets = np.load(bet_loc)
            if weight_loc:
                self.weights = np.load(weight_loc)
            if value_loc:
                self.values = np.load(value_loc)
            return True
        except FileNotFoundError:
            return False

    def save_to_disk(self, sample_count_loc, obs_loc, obs_count_loc, action_loc, bet_loc, weight_loc=None, value_loc=None):
        np.save(sample_count_loc, self.sample_count)
        np.save(obs_loc, self.obs)
        np.save(obs_count_loc, self.obs_count)
        np.save(action_loc, self.actions)
        np.save(bet_loc, self.bets)
        if weight_loc:
            np.save(weight_loc, self.weights)
        if value_loc:
            np.save(value_loc, self.values)

    def add(self, obs, obs_counts, actions, bets, weights=None, values=None):
        count = self.sample_count[0]
        if count + obs.shape[0] < self.size:
            self.obs[count:count + obs.shape[0]] = obs
            self.obs_count[count:count + obs.shape[0]] = obs_counts
            self.actions[count:count + obs.shape[0]] = actions
            self.bets[count:count + obs.shape[0]] = bets
            if weights is not None:
                self.weights[count:count + obs.shape[0]] = weights
            if values is not None:
                self.values[count:count + obs.shape[0]] = values
            self.sample_count += obs.shape[0]
        else:
            should_replace = (np.random.uniform(0, 1, obs.shape[0]) > (1 - self.size / count)).astype(np.int32)
            obs = obs[should_replace.nonzero()]
            obs_counts = obs_counts[should_replace.nonzero()]
            actions = actions[should_replace.nonzero()]
            bets = bets[should_replace.nonzero()]
            replace_indices = np.random.choice(np.arange(self.size), np.count_nonzero(should_replace))
            replace_indices, sample_indices = np.unique(replace_indices, return_index=True)
            self.obs[replace_indices] = obs[sample_indices]
            self.obs_count[replace_indices] = obs_counts[sample_indices]
            self.actions[replace_indices] = actions[sample_indices]
            self.bets[replace_indices] = bets[sample_indices]
            if weights is not None:
                self.weights[replace_indices] = weights[sample_indices]
            if values is not None:
                self.values[replace_indices] = values[sample_indices]
        if self.new_batch_event is not None:
            self.new_data_event.set()
        if self.new_batch_event is not None:
            self.total_fresh_samples += len(obs)
            if self.total_fresh_samples >= self.batch_size:
                self.new_batch_event.set()
