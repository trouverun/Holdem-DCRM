import numpy as np
import random
from pokerenv.common import Action, PlayerAction, action_list
from config import SEQUENCE_LENGTH, OBS_SHAPE, N_PLAYERS, EVAL_HH_FREQUENCY, BET_BUCKETS, N_ACTIONS, N_BET_BUCKETS
import pokerenv.obs_indices as indices
import logging


class BatchedEnvironment:
    def __init__(self, create_env_fn, batch_size):
        self.envs = [create_env_fn() for _ in range(batch_size)]
        self.batch_size = batch_size
        self.should_disable_hh = np.zeros(batch_size)
        self.observations = np.zeros([batch_size, N_PLAYERS, SEQUENCE_LENGTH, OBS_SHAPE], dtype=np.float32)
        self.observation_counts = np.zeros([batch_size, N_PLAYERS], dtype=np.int32)
        self.tracked_player_rewards = np.zeros([batch_size])
        self.current_acting_players = np.zeros(batch_size)
        self.valid_actions = np.zeros([batch_size, N_ACTIONS])
        self.current_bet_limits = np.zeros([batch_size, 2])
        self.current_pot_sizes = np.zeros(batch_size)

    def reset(self):
        inference_mappings = dict()
        inference_observations = [[] for _ in range(N_PLAYERS)]
        inference_observation_counts = [[] for _ in range(N_PLAYERS)]
        for env in range(self.batch_size):
            if random.uniform(0, 1) < 1 / EVAL_HH_FREQUENCY:
                self.envs[env].hand_history_enabled = True
                self.should_disable_hh[env] = 1
            obs = self.envs[env].reset()
            next_acting_player = int(obs[indices.ACTING_PLAYER])
            self.current_acting_players[env] = next_acting_player
            self.current_pot_sizes[env] = obs[indices.POT_SIZE]
            self.valid_actions[env] = obs[indices.VALID_ACTIONS]
            self.current_bet_limits[env] = np.array([obs[indices.VALID_BET_LOW], obs[indices.VALID_BET_HIGH]])
            self.observation_counts[env, next_acting_player] = 1
            self.observations[env, next_acting_player, 0, :] = obs

            inference_observations[next_acting_player].append(self.observations[env, next_acting_player])
            inference_observation_counts[next_acting_player].append(1)
            inference_mappings[env] = (next_acting_player, len(inference_observations[next_acting_player]) - 1)

        return inference_observations, inference_observation_counts, inference_mappings

    def step(self, action_preds, bet_preds, inference_mappings):
        new_inference_mappings = dict()
        inference_observations = [[] for _ in range(N_PLAYERS)]
        inference_observation_counts = [[] for _ in range(N_PLAYERS)]
        rewards = []

        for env in range(self.batch_size):
            current_actor = self.current_acting_players[env]
            bet_array = np.concatenate([
                np.array([self.current_bet_limits[env, 0], self.current_bet_limits[env].sum()/2, self.current_bet_limits[env, 1]]),
                BET_BUCKETS * self.current_pot_sizes[env]
            ])
            p_action = np.exp(action_preds[inference_mappings[env][0]][inference_mappings[env][1]])
            p_bet = np.exp(bet_preds[inference_mappings[env][0]][inference_mappings[env][1]])
            if current_actor != 5:
                p_action = np.zeros(N_ACTIONS)
                p_action[np.argwhere(self.valid_actions[env] == 1)] = 1 / np.count_nonzero(self.valid_actions[env])
                min_bet = self.current_bet_limits[env, 0]
                max_bet = self.current_bet_limits[env, 1]
                bet_sizes = np.concatenate(
                    [np.array([min_bet, (min_bet + max_bet) / 2, max_bet]), BET_BUCKETS * self.current_pot_sizes[env]])
                invalid_bets = np.logical_or(min_bet > bet_sizes, bet_sizes > max_bet)
                p_bet = np.zeros(N_BET_BUCKETS)
                p_bet[np.logical_not(invalid_bets)] = 1 / np.count_nonzero(np.logical_not(invalid_bets))
            action = np.random.choice(action_list, p=p_action)
            action = action_list[action]
            bet = np.random.choice(bet_array, p=p_bet)
            table_action = Action(action, bet)
            obs, reward, env_done, _ = self.envs[env].step(table_action)
            if current_actor == 0:
                self.tracked_player_rewards[env] += reward
                if env_done or ((obs[indices.HAND_IS_OVER] or obs[indices.DELAYED_REWARD])):
                    rewards.append(self.tracked_player_rewards[env])
                    self.observations[env, :, :, :] = np.zeros(OBS_SHAPE, dtype=np.float32)
                    self.observation_counts[env, :] = np.zeros(N_PLAYERS, dtype=np.int32)
                    self.tracked_player_rewards[env] = 0
                    if self.should_disable_hh[env] == 1:
                        self.should_disable_hh[env] = 0
                        self.envs[env].hand_history_enabled = False
                    if random.uniform(0, 1) < 1 / EVAL_HH_FREQUENCY:
                        self.envs[env].hand_history_enabled = True
                        self.should_disable_hh[env] = 1
                    obs = self.envs[env].reset()
            next_acting_player = int(obs[indices.ACTING_PLAYER])
            self.current_acting_players[env] = next_acting_player
            self.current_pot_sizes[env] = obs[indices.POT_SIZE]
            self.valid_actions[env] = obs[indices.VALID_ACTIONS]
            self.current_bet_limits[env] = np.array([obs[indices.VALID_BET_LOW], obs[indices.VALID_BET_HIGH]])
            obs_count = min(SEQUENCE_LENGTH, self.observation_counts[env, next_acting_player] + 1)
            self.observation_counts[env, next_acting_player] = obs_count
            if obs_count == SEQUENCE_LENGTH:
                self.observations[env, next_acting_player, 0:SEQUENCE_LENGTH - 1, :] = self.observations[env, next_acting_player, 1:SEQUENCE_LENGTH, :]
                self.observations[env, next_acting_player, -1, :] = obs
            else:
                self.observations[env, next_acting_player, obs_count-1, :] = obs
            inference_observations[next_acting_player].append(self.observations[env, next_acting_player])
            inference_observation_counts[next_acting_player].append(obs_count)
            new_inference_mappings[env] = (next_acting_player, len(inference_observations[next_acting_player]) - 1)

        return inference_observations, inference_observation_counts, new_inference_mappings, np.asarray(rewards)


