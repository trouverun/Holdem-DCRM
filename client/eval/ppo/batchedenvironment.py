import numpy as np
from math import inf
import pokerenv.obs_indices as indices
from random import shuffle as rshuffle
from copy import deepcopy
from config import N_PLAYERS, MAX_EPISODE_LENGTH, OBS_SHAPE, SEQUENCE_LENGTH, BET_BUCKETS, DISCOUNT_RATE, N_ACTIONS, N_BET_BUCKETS, \
    N_PPO_EVAL_HANDS, EVAL_CONSIDER_SINGLE_TRAJECTORY, N_LOGGED_HANDS
from pokerenv.common import Action, PlayerAction, action_list


class BatchedEnvironment:
    def __init__(self, batch_size, ppo_player, trajectory_que, evaluated_player_reward_que):
        self.training_mode = True
        self.initial_env = None
        self.initial_obs = None
        self.active_observations = np.zeros([batch_size, OBS_SHAPE])
        self.observations = np.zeros([batch_size, N_PLAYERS, MAX_EPISODE_LENGTH, OBS_SHAPE], dtype=np.float32)
        self.observation_counts = np.zeros([batch_size, N_PLAYERS], dtype=np.int32)
        self.action_probabilities = np.zeros([batch_size, N_PLAYERS, MAX_EPISODE_LENGTH, 2], dtype=np.float32)  # chosen action_i, chosen action_log_prob
        self.bet_probabilities = np.zeros([batch_size, N_PLAYERS, MAX_EPISODE_LENGTH, 2], dtype=np.float32)     # chosen bet_i, chosen bet_log_prob
        self.final_rewards = np.zeros([batch_size, N_PLAYERS])
        self.envs = []
        self.batch_size = batch_size
        self.trajectory_que = trajectory_que
        self.ppo_player = ppo_player
        self.evaluated_player_reward_que = evaluated_player_reward_que
        self.enabled_envs = set(range(batch_size))

    def _append_obs(self, env_i, acting_player, obs):
        self.active_observations[env_i] = obs
        if self.training_mode:
            self.observation_counts[env_i, acting_player] += 1
            observation_count = self.observation_counts[env_i, acting_player]
            self.observations[env_i, acting_player, observation_count - 1] = obs
            if observation_count >= SEQUENCE_LENGTH:
                return self.observations[env_i, acting_player, observation_count - SEQUENCE_LENGTH:observation_count], min(SEQUENCE_LENGTH, observation_count)
            else:
                return self.observations[env_i, acting_player, 0:SEQUENCE_LENGTH], min(SEQUENCE_LENGTH, observation_count)
        else:
            observation_count = min(SEQUENCE_LENGTH, self.observation_counts[env_i, acting_player]+1)
            self.observation_counts[env_i, acting_player] = observation_count
            if observation_count == SEQUENCE_LENGTH:
                self.observations[env_i, acting_player, 0:SEQUENCE_LENGTH-1] = self.observations[env_i, acting_player, 1:SEQUENCE_LENGTH]
                self.observations[env_i, acting_player, -1] = obs
                return self.observations[env_i, acting_player], observation_count
            else:
                self.observations[env_i, acting_player, observation_count-1] = obs
                return self.observations[env_i, acting_player], observation_count

    def _get_table_action(self, env_i, obs, acting_player, action_p, bet_p):
        min_bet = obs[indices.VALID_BET_LOW]
        max_bet = obs[indices.VALID_BET_HIGH]
        bet_sizes = np.concatenate(
            [np.array([min_bet, (min_bet + max_bet) / 2, max_bet]), BET_BUCKETS * obs[indices.POT_SIZE]]
        )
        action = np.random.choice(action_list, p=np.exp(action_p))
        player_action = action_list[action]
        bet_bucket = 0
        bet_size = 0
        if player_action == PlayerAction.BET:
            bet_bucket = np.random.choice(range(N_BET_BUCKETS), p=np.exp(bet_p))
            bet_size = bet_sizes[bet_bucket]

        # For the trained best response actors, record the probabilities and selections for importance sampling later on
        if acting_player == self.ppo_player and self.training_mode:
            action_log_prob = action_p[action]
            if player_action == PlayerAction.BET:
                bet_log_prob = bet_p[bet_bucket]
            else:
                bet_log_prob = 0
            obs_count = self.observation_counts[env_i, acting_player]
            self.action_probabilities[env_i, acting_player, obs_count - 1] = np.array([action, action_log_prob])
            self.bet_probabilities[env_i, acting_player, obs_count - 1] = np.array([bet_bucket, bet_log_prob])
        return Action(player_action, bet_size)

    def _reset_env(self, env_i):
        self.observation_counts[env_i] = np.zeros(N_PLAYERS, dtype=np.int32)
        if self.training_mode:
            self.observations[env_i] = np.zeros([N_PLAYERS, MAX_EPISODE_LENGTH, OBS_SHAPE], dtype=np.float32)
            self.action_probabilities[env_i] = np.zeros([N_PLAYERS, MAX_EPISODE_LENGTH, 2], dtype=np.float32)
            self.bet_probabilities[env_i] = np.zeros([N_PLAYERS, MAX_EPISODE_LENGTH, 2], dtype=np.float32)
            self.final_rewards[env_i] = np.zeros(N_PLAYERS, dtype=np.float32)
        else:
            self.observations[env_i] = np.zeros([N_PLAYERS, SEQUENCE_LENGTH, OBS_SHAPE], dtype=np.float32)
            self.final_rewards[env_i] = 0
        self.envs[env_i] = deepcopy(self.initial_env)
        if not EVAL_CONSIDER_SINGLE_TRAJECTORY:
            rshuffle(self.envs[env_i].deck.cards)
        if not self.training_mode:
            if np.random.uniform() > 1 - (N_LOGGED_HANDS / N_PPO_EVAL_HANDS):
                self.envs[env_i].hand_history_enabled = True
        acting_player = int(self.initial_obs[indices.ACTING_PLAYER])
        obs_sequence, obs_count = self._append_obs(env_i, acting_player, self.initial_obs)
        return obs_sequence, obs_count, acting_player

    def _finish_env(self, env_i):
        if self.training_mode:
            for player_i in range(N_PLAYERS):
                if player_i == self.ppo_player:
                    observations = []
                    obs_counts = []
                    action_probabilities = []
                    bet_probabilities = []
                    rewards = []
                    player_observation_count = self.observation_counts[env_i, player_i]
                    idxs = np.arange(player_observation_count)
                    discounts = np.power(DISCOUNT_RATE, idxs)
                    for obs_count in range(1, player_observation_count+1):
                        start_i = max(0, obs_count - SEQUENCE_LENGTH)
                        n_select = min(obs_count, SEQUENCE_LENGTH)
                        data = self.observations[env_i, player_i, start_i:start_i + n_select]
                        if obs_count < SEQUENCE_LENGTH:
                            padding = np.zeros([SEQUENCE_LENGTH - obs_count, OBS_SHAPE])
                            data = np.concatenate([data, padding])
                        observations.append(data)
                        action_probabilities.append(self.action_probabilities[env_i, player_i, obs_count])
                        bet_probabilities.append(self.bet_probabilities[env_i, player_i, obs_count])
                        rewards.append(discounts[obs_count-1]*self.final_rewards[env_i, player_i])
                        obs_counts.append(SEQUENCE_LENGTH if obs_count >= SEQUENCE_LENGTH else obs_count)
                    self.trajectory_que.put((observations, obs_counts, action_probabilities, bet_probabilities, rewards))
        else:
            self.evaluated_player_reward_que.put(self.final_rewards[env_i].mean())
        return self._reset_env(env_i)

    def set_training_mode(self, value: bool):
        self.training_mode = value
        if value:
            self.active_observations = np.zeros([self.batch_size, OBS_SHAPE])
            self.observations = np.zeros([self.batch_size, N_PLAYERS, MAX_EPISODE_LENGTH, OBS_SHAPE], dtype=np.float32)
            self.observation_counts = np.zeros([self.batch_size, N_PLAYERS], dtype=np.int32)
            self.action_probabilities = np.zeros([self.batch_size, N_PLAYERS, MAX_EPISODE_LENGTH, 2], dtype=np.float32)  # chosen action_i, chosen action_log_prob
            self.bet_probabilities = np.zeros([self.batch_size, N_PLAYERS, MAX_EPISODE_LENGTH, 2], dtype=np.float32)     # chosen bet_i, chosen bet_log_prob
            self.final_rewards = np.zeros([self.batch_size, N_PLAYERS])
        else:
            self.active_observations = np.zeros([self.batch_size, OBS_SHAPE])
            self.observations = np.zeros([self.batch_size, N_PLAYERS, SEQUENCE_LENGTH, OBS_SHAPE], dtype=np.float32)
            self.observation_counts = np.zeros([self.batch_size, N_PLAYERS], dtype=np.int32)
            self.action_probabilities = None
            self.bet_probabilities = None
            self.final_rewards = np.zeros([self.batch_size, N_PLAYERS])

    def set_env(self, env, initial_obs):
        self.initial_env = env
        self.initial_obs = initial_obs
        for env_i in range(self.batch_size):
            self.envs.append(deepcopy(env))

    def reset(self):
        self.enabled_envs = set(range(self.batch_size))
        inference_observations = [[] for _ in range(N_PLAYERS)]
        inference_observation_counts = [[] for _ in range(N_PLAYERS)]
        mappings = dict()
        for env_i in self.enabled_envs:
            obs_sequence, obs_count, acting_player = self._reset_env(env_i)
            inference_observations[acting_player].append(obs_sequence)
            inference_observation_counts[acting_player].append(obs_count)
            mappings[env_i] = [acting_player, len(inference_observations[acting_player])-1]
        return inference_observations, inference_observation_counts, mappings, list(self.enabled_envs)

    def step(self, action_preds, bet_preds, mappings):
        inference_observations = [[] for _ in range(N_PLAYERS)]
        inference_observation_counts = [[] for _ in range(N_PLAYERS)]
        new_mappings = dict()
        reset_envs = []
        for env_i in self.enabled_envs:
            table = self.envs[env_i]
            if env_i in mappings.keys():
                ap = action_preds[mappings[env_i][0]][mappings[env_i][1]]
                bp = bet_preds[mappings[env_i][0]][mappings[env_i][1]]
            else:
                # If env_i is not in the mappings keys,
                # the env is asking for don't care actions to flush out final rewards for each remaining player
                ap = -inf*np.ones(N_ACTIONS)
                ap[0] = 0
                bp = np.zeros(N_BET_BUCKETS)
            obs = self.active_observations[env_i]
            previous_acting_player = int(obs[indices.ACTING_PLAYER])
            action = self._get_table_action(env_i, obs, previous_acting_player, ap, bp)
            obs, reward, env_done, info = table.step(action)

            if self.training_mode:
                done = env_done
            else:
                done = False

            # Only record observations if they correspond to a valid decision-making point,
            # rather than final reward collection with don't care actions
            if not obs[indices.HAND_IS_OVER]:
                acting_player = int(obs[indices.ACTING_PLAYER])
                obs_sequence, obs_count = self._append_obs(env_i, acting_player, obs)
                inference_observations[acting_player].append(obs_sequence)
                inference_observation_counts[acting_player].append(obs_count)
                new_mappings[env_i] = [acting_player, len(inference_observations[acting_player]) - 1]
                continue
            else:
                self.active_observations[env_i] = obs
                if self.training_mode:
                    self.final_rewards[env_i, previous_acting_player] = reward
                else:
                    if previous_acting_player != self.ppo_player:
                        self.final_rewards[env_i, previous_acting_player] = -reward
                        done = True
            if done:
                obs_sequence, obs_count, acting_player = self._finish_env(env_i)
                inference_observations[acting_player].append(obs_sequence)
                inference_observation_counts[acting_player].append(obs_count)
                new_mappings[env_i] = [acting_player, len(inference_observations[acting_player]) - 1]
                reset_envs.append(env_i)

        return inference_observations, inference_observation_counts, new_mappings, reset_envs

    def disable_envs(self, envs):
        for env_i in envs:
            self.enabled_envs.remove(env_i)
        return len(self.enabled_envs)