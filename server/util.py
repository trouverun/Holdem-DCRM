import numpy as np
import torch
import logging
from config import MAX_TRAIN_BATCH_SIZE, OBS_SHAPE, N_BET_BUCKETS, N_EPOCHS, PATIENCE, SEQUENCE_LENGTH, N_ACTIONS, RESERVOIR_SIZE


class Learner:
    def __init__(self, device):
        self.device = device

    def _training_loop(self, net, optim, loss_fn, obs, obs_counts, actions, bets, train_indices, validation_indices, scheduler, step_point, iteration=None, weights=None):
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
                # Discounting in the case of linear CFR
                if weights is not None:
                    batch_weights = torch.from_numpy(weights[batch_indices]).type(torch.LongTensor).to(self.device)
                    y_action = 2/iteration * batch_weights * y_action
                    y_bet = 2/iteration * batch_weights * y_bet
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


class Reservoir:
    def __init__(self):
        self.sample_count = np.zeros(1, dtype=np.int64)
        self.obs = np.zeros([RESERVOIR_SIZE, SEQUENCE_LENGTH, OBS_SHAPE], dtype=np.float32)
        self.obs_count = np.zeros([RESERVOIR_SIZE, 1])
        self.actions = np.zeros([RESERVOIR_SIZE, N_ACTIONS], dtype=np.float32)
        self.bets = np.zeros([RESERVOIR_SIZE, N_BET_BUCKETS], dtype=np.float32)
        self.weights = np.zeros([RESERVOIR_SIZE, 1])

    def load_from_disk(self, sample_count_loc, obs_loc, obs_count_loc, action_loc, bet_loc):
        try:
            self.sample_count = np.load(sample_count_loc)
            self.obs = np.load(obs_loc).reshape(RESERVOIR_SIZE, SEQUENCE_LENGTH, OBS_SHAPE)
            self.obs_count = np.load(obs_count_loc)
            self.actions = np.load(action_loc)
            self.bets = np.load(bet_loc)
            return True
        except FileNotFoundError:
            return False

    def save_to_disk(self, sample_count_loc, obs_loc, obs_count_loc, action_loc, bet_loc):
        np.save(sample_count_loc, self.sample_count)
        np.save(obs_loc, self.obs)
        np.save(obs_count_loc, self.obs_count)
        np.save(action_loc, self.actions)
        np.save(bet_loc, self.bets)

    def add(self, obs, obs_counts, actions, bets, weights=None):
        count = self.sample_count[0]
        if count + obs.shape[0] < RESERVOIR_SIZE:
            self.obs[count:count + obs.shape[0]] = obs
            self.obs_count[count:count + obs.shape[0]] = obs_counts
            self.actions[count:count + obs.shape[0]] = actions
            self.bets[count:count + obs.shape[0]] = bets
            if weights is not None:
                self.weights[count:count + obs.shape[0]] = weights
            self.sample_count += obs.shape[0]
        else:
            should_replace = (np.random.uniform(0, 1, obs.shape[0]) > (1 - RESERVOIR_SIZE / count)).astype(np.int32)
            obs = obs[should_replace.nonzero()]
            obs_counts = obs_counts[should_replace.nonzero()]
            actions = actions[should_replace.nonzero()]
            bets = bets[should_replace.nonzero()]
            replace_indices = np.random.choice(np.arange(RESERVOIR_SIZE), np.count_nonzero(should_replace))
            replace_indices, sample_indices = np.unique(replace_indices, return_index=True)
            self.obs[replace_indices] = obs[sample_indices]
            self.obs_count[replace_indices] = obs_counts[sample_indices]
            self.actions[replace_indices] = actions[sample_indices]
            self.bets[replace_indices] = bets[sample_indices]
            if weights is not None:
                self.weights[replace_indices] = weights[sample_indices]

