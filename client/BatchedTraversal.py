import copy
import numpy as np
import pokerenv.obs_indices as indices
from pokerenv.table import Table
from pokerenv.common import Action, PlayerAction, action_list
from config import N_PLAYERS, LOW_STACK_BBS, HIGH_STACK_BBS, HH_LOCATION, INVALID_ACTION_PENALTY, OBS_SHAPE, SEQUENCE_LENGTH, \
    N_BET_BUCKETS, BET_BUCKETS, N_ACTIONS


# Container class for multiple (deep) counterfactual regret minimization traversals
class BatchedTraversal:
    def __init__(self, n_traversals, traverser, regret_que, strategy_que):
        assert 0 <= traverser < N_PLAYERS
        self.n_traversals = n_traversals
        self.traverser = traverser
        self.regret_que = regret_que
        self.strategy_que = strategy_que
        self.node_n = 0
        self.all_nodes = dict()
        # Nodes that need to have their predicted regrets added, and possible child nodes expanded
        self.active_nodes = set()

    def reset(self):
        self.all_nodes.clear()
        self.active_nodes.clear()
        self.node_n = 0
        # Group the observation data by player, so that we can batch them for the inference server(s)
        inference_observations = [[] for _ in range(N_PLAYERS)]
        inference_observation_counts = [[] for _ in range(N_PLAYERS)]
        # Mapping from key -> [row,col], which allows us to later retrieve the regrets, which are also grouped by player
        mappings = dict()
        # Create root nodes for each traversal
        for _ in range(self.n_traversals):
            env = Table(N_PLAYERS, LOW_STACK_BBS, HIGH_STACK_BBS, HH_LOCATION, INVALID_ACTION_PENALTY)
            self.all_nodes[self.node_n] = Node(self.node_n, None, env)
            node = self.all_nodes[self.node_n]
            self.active_nodes.add(self.node_n)
            initial_observations = node.reset()
            inference_observations[node.acting_player].append(initial_observations)
            inference_observation_counts[node.acting_player].append(1)
            # Create mapping key -> [row, col], which we can use later to retrieve the corresponding regrets
            mappings[self.node_n] = [node.acting_player, len(inference_observations[node.acting_player])-1]
            self.node_n += 1
        return inference_observations, inference_observation_counts, mappings

    # When a leaf node is reached by the traverser, this function propagates the reward upward,
    # until it reaches the root node, or another traverser node that is still waiting for its other child nodes to return
    def _propagate_reward(self, node, reward):
        while True:
            parent = node.parent
            if parent is None:
                break
            # If action_type is None, there is no expected value to calculate, and we can directly propagate reward one node further
            was_action_dont_care = node.action_type is None
            if parent.acting_player == self.traverser and not was_action_dont_care:
                if node.bet_bucket is not None:
                    parent.rewards_bet[node.bet_bucket] = reward
                    parent.n_rewards_bet -= 1
                    if parent.n_rewards_bet == 0:
                        parent.rewards_action[2] = (parent.pi_bet * parent.rewards_bet).sum()
                        parent.n_rewards_action -= 1
                else:
                    parent.rewards_action[int(node.action_type)] = reward
                    parent.n_rewards_action -= 1
                if parent.n_rewards_action == 0:
                    expected_reward = (parent.pi_action * parent.rewards_action).sum()
                    action_regrets = parent.rewards_action - expected_reward
                    bet_expected_reward = parent.rewards_action[2]
                    bet_regrets = parent.rewards_bet - bet_expected_reward
                    self.regret_que.put((parent.observation_history, parent.observation_count, action_regrets, bet_regrets))
                    self._propagate_reward(parent, expected_reward)
                    self.all_nodes.pop(node.id)
                break
            else:
                self.all_nodes.pop(node.id)
                node = parent

    def _create_child_node(self, parent, action, bet, observations, observation_counts, mappings, prev_obs):
        env = parent.env
        env_copy = copy.deepcopy(env)
        self.all_nodes[self.node_n] = Node(self.node_n, parent, env_copy, action, bet)
        node = self.all_nodes[self.node_n]
        obs, obs_count, reward, done = node.step(prev_obs, parent_is_traverser=True)
        if not done:
            self.active_nodes.add(self.node_n)
            observations[node.acting_player].append(obs)
            observation_counts[node.acting_player].append(obs_count)
            mappings[self.node_n] = [node.acting_player, len(observations[node.acting_player]) - 1]
        else:
            self._propagate_reward(node, reward)
        self.node_n += 1

    def _create_child_nodes(self, obs, parent, observations, observation_counts, mappings):
        valid_actions = obs[indices.VALID_ACTIONS]
        min_bet = obs[indices.VALID_BET_LOW]
        max_bet = obs[indices.VALID_BET_HIGH]
        bet_sizes = np.concatenate(
            [np.array([min_bet, (min_bet + max_bet) / 2, max_bet]), BET_BUCKETS * obs[indices.POT_SIZE]]
        )
        valid_bets = np.logical_and(min_bet <= bet_sizes, bet_sizes <= max_bet)

        parent.n_rewards_action = len(np.argwhere(valid_actions == 1))
        parent.n_rewards_bet = len(np.argwhere(valid_bets))
        for action in np.argwhere(valid_actions == 1):
            action = np.squeeze(action)
            if action_list[action] == PlayerAction.BET:
                for bet in np.argwhere(valid_bets):
                    self._create_child_node(parent, action_list[action], bet, observations, observation_counts, mappings, obs)
            else:
                self._create_child_node(parent, action_list[action], None, observations, observation_counts, mappings, obs)

    def step(self, action_regrets, bet_regrets, mappings):
        observations = [[] for _ in range(N_PLAYERS)]
        observation_counts = [[] for _ in range(N_PLAYERS)]
        new_mappings = dict()
        # Iterate over all active nodes and:
        #   1) add predicted regrets to the current node
        #   2) sample child nodes with external sampling
        for key in list(self.active_nodes):
            self.active_nodes.remove(key)
            node = self.all_nodes[key]
            ar = action_regrets[mappings[key][0]][mappings[key][1]]
            br = bet_regrets[mappings[key][0]][mappings[key][1]]
            previous_observation = node.observation_history[node.observation_count-1]
            acting_player = previous_observation[indices.ACTING_PLAYER]
            # If the hand is over, we are only feeding junk actions to get the end of hand rewards back for each player
            action_dont_care = previous_observation[indices.DELAYED_REWARD]
            node.add_regrets(previous_observation, ar, br)
            parent_is_traverser = False
            if acting_player == self.traverser:
                # If action is "don't care" there is no real choice, and thus no need to explore child nodes
                if not action_dont_care:
                    # Expand all possible trajectories/child nodes originating from the current node
                    self._create_child_nodes(previous_observation, node, observations, observation_counts, new_mappings)
                    node.env = None  # Save some memory by removing (now) unused environment
                    continue
                else:
                    parent_is_traverser = True
            env = node.env
            env_copy = copy.deepcopy(env)
            node.env = None  # Save some memory by removing (now) unused environment
            self.all_nodes[self.node_n] = Node(self.node_n, node, env_copy)
            new_node = self.all_nodes[self.node_n]
            obs, obs_count, reward, done = new_node.step(previous_observation, parent_is_traverser, action_dont_care)
            if not parent_is_traverser:
                self.strategy_que.put((node.observation_history, obs_count, node.pi_action, node.pi_bet))
            if not done:
                self.active_nodes.add(self.node_n)
                observations[new_node.acting_player].append(obs)
                observation_counts[new_node.acting_player].append(obs_count)
                new_mappings[self.node_n] = [new_node.acting_player, len(observations[new_node.acting_player]) - 1]
            else:
                self._propagate_reward(new_node, reward)
            self.node_n += 1
        return observations, observation_counts, new_mappings


# Represents a single decision point in a trajectory traversal
class Node:
    def __init__(self, id, parent, env, action_type=None, bet_bucket=None):
        self.id = id
        self.parent = parent
        self.env = env
        self.acting_player = None
        # The observation history, under which the action leading to the next node was taken
        self.observation_history = np.zeros([SEQUENCE_LENGTH, OBS_SHAPE], dtype=np.float32)
        self.observation_count = 0
        # Which action was applied to the parent environment to reach this node (used to fill the parent reward array)
        self.action_type = action_type
        # Which bet was applied to the parent environment to reach this node (used to fill the parent reward array)
        self.bet_bucket = bet_bucket
        self.rewards_action = np.zeros(N_ACTIONS, dtype=np.float32)
        self.rewards_bet = np.zeros(N_BET_BUCKETS, dtype=np.float32)
        self.n_rewards_action = np.zeros(N_ACTIONS)
        self.n_rewards_bet = np.zeros(N_BET_BUCKETS)
        self.pi_action = np.zeros(N_ACTIONS, dtype=np.float32)
        self.pi_bet = np.zeros(N_BET_BUCKETS, dtype=np.float32)

    def reset(self):
        obs = self.env.reset()
        self.acting_player = int(obs[indices.ACTING_PLAYER])
        self.observation_history[0] = obs
        self.observation_count += 1
        return self.observation_history

    def add_regrets(self, previous_obs, action_regrets, bet_regrets):
        valid_actions = previous_obs[indices.VALID_ACTIONS]
        min_bet = previous_obs[indices.VALID_BET_LOW]
        max_bet = previous_obs[indices.VALID_BET_HIGH]
        bet_sizes = np.concatenate(
            [np.array([min_bet, (min_bet + max_bet) / 2, max_bet]), BET_BUCKETS * previous_obs[indices.POT_SIZE]])
        invalid_bets = np.logical_or(min_bet > bet_sizes, bet_sizes > max_bet)

        # If all probabilities are negative select the least negative entry deterministically like in the paper
        if np.count_nonzero(action_regrets >= 0) == 0:
            chosen_i = action_regrets.argmax()
            action_regrets[chosen_i] = 1
        action_regrets = np.maximum(action_regrets, np.zeros_like(action_regrets), dtype=np.float32)
        # Mask invalid actions and make new probabilities sum to 1
        if np.count_nonzero(action_regrets) == 0:
            self.pi_action[np.argwhere(valid_actions == 1)] = 1 / np.count_nonzero(valid_actions)
        else:
            self.pi_action = action_regrets / action_regrets.sum()

        # If all probabilities are negative select the least negative entry deterministically like in the paper
        if np.count_nonzero(bet_regrets >= 0) == 0:
            chosen_i = bet_regrets.argmax()
            bet_regrets[chosen_i] = 1
        bet_regrets = np.maximum(bet_regrets, np.zeros_like(bet_regrets), dtype=np.float32)
        # Mask invalid bets and make new probabilities sum to 1
        if np.count_nonzero(bet_regrets) == 0:
            self.pi_bet[np.logical_not(invalid_bets)] = 1 / np.count_nonzero(np.logical_not(invalid_bets))
        else:
            self.pi_bet = bet_regrets / bet_regrets.sum()

    def step(self, previous_obs, parent_is_traverser=False, action_dont_care=False):
        table_action = Action(PlayerAction.CHECK, 0)
        # If action is a "don't care" we are only feeding junk actions to the environment in order to get the final rewards back
        if not action_dont_care:
            min_bet = previous_obs[indices.VALID_BET_LOW]
            max_bet = previous_obs[indices.VALID_BET_HIGH]
            bet_sizes = np.concatenate(
                [np.array([min_bet, (min_bet + max_bet) / 2, max_bet]), BET_BUCKETS * previous_obs[indices.POT_SIZE]])
            # If action type is None, then the acting player is not traverser, and we sample the action from the parent distribution
            if not parent_is_traverser:
                action = np.random.choice(action_list, p=self.parent.pi_action)
                action = action_list[action]
                bet_size = 0
                if action == PlayerAction.BET:
                    bet_size = np.random.choice(bet_sizes, p=self.parent.pi_bet)
                table_action = Action(action, bet_size)
            else:
                bet_size = 0
                if self.bet_bucket is not None:
                    bet_size = bet_sizes[self.bet_bucket][0]
                table_action = Action(self.action_type, bet_size)
        obs, reward, env_done, info = self.env.step(table_action)
        done = env_done or ((obs[indices.HAND_IS_OVER] or action_dont_care) and parent_is_traverser)
        if not done:
            # Construct observation history for this node
            self.observation_history = np.zeros([SEQUENCE_LENGTH, OBS_SHAPE], dtype=np.float32)
            self.observation_history[0] = obs
            self.observation_count = 1
            next_player = int(obs[indices.ACTING_PLAYER])
            self.acting_player = next_player
            p_parent = self.parent
            while True:
                if p_parent is None:
                    break
                if p_parent.acting_player == next_player:
                    self.observation_count = min(p_parent.observation_count + 1, SEQUENCE_LENGTH)
                    if p_parent.observation_count == SEQUENCE_LENGTH:
                        self.observation_history[0:SEQUENCE_LENGTH - 1] = p_parent.observation_history[1:SEQUENCE_LENGTH]
                        self.observation_history[-1] = obs
                    else:
                        self.observation_history = p_parent.observation_history.copy()
                        self.observation_history[self.observation_count - 1] = obs
                    break
                p_parent = p_parent.parent
        return self.observation_history, self.observation_count, reward, done
