import copy
import gc
import logging
import numpy as np
import pokerenv.obs_indices as indices
from pokerenv.table import Table
from pokerenv.common import Action, PlayerAction, action_list
from config import N_PLAYERS, LOW_STACK_BBS, HIGH_STACK_BBS, HH_LOCATION, INVALID_ACTION_PENALTY, OBS_SHAPE, SEQUENCE_LENGTH, \
    N_BET_BUCKETS, BET_BUCKETS, N_ACTIONS


class BatchedTraversal:
    def __init__(self, n_traversals, traverser, regret_que, strategy_ques):
        assert 0 <= traverser < N_PLAYERS
        self.n_traversals = n_traversals
        self.traverser = traverser
        self.regret_que = regret_que
        self.strategy_ques = strategy_ques
        self.node_n = 0
        self.all_nodes = dict()
        # Set of nodes that need to have their predicted regrets added, and possible child nodes expanded
        self.waiting_nodes = set()

        self.regret_obs_list = []
        self.regret_obs_count_list = []
        self.regret_action_list = []
        self.regret_bet_list = []
        self.strategy_obs_list = [[] for _ in range(N_PLAYERS)]
        self.strategy_obs_count_list = [[] for _ in range(N_PLAYERS)]
        self.strategy_action_list = [[] for _ in range(N_PLAYERS)]
        self.strategy_bet_list = [[] for _ in range(N_PLAYERS)]

    def reset(self):
        self.all_nodes.clear()
        self.waiting_nodes.clear()
        self.node_n = 0
        # Group the initial observation data by the acting player, so that we can batch them for the inference server(s)
        inference_observations = [[] for _ in range(N_PLAYERS)]
        inference_observation_counts = [[] for _ in range(N_PLAYERS)]
        mappings = dict()
        # Create root nodes for each traversal
        for _ in range(self.n_traversals):
            env = Table(N_PLAYERS, LOW_STACK_BBS, HIGH_STACK_BBS, HH_LOCATION, INVALID_ACTION_PENALTY)
            self.all_nodes[self.node_n] = Node(self.node_n, None, env)
            node = self.all_nodes[self.node_n]
            self.waiting_nodes.add(self.node_n)
            initial_observations = node.reset()
            inference_observations[node.acting_player].append(initial_observations)
            inference_observation_counts[node.acting_player].append(1)
            # Create the mapping key -> [row, col], which allows us to later retrieve the corresponding predicted regrets
            mappings[self.node_n] = [node.acting_player, len(inference_observations[node.acting_player])-1]
            self.node_n += 1
        return inference_observations, inference_observation_counts, mappings

    # When a leaf node is reached by the traverser, this function propagates the reward upward,
    # until it reaches the root node, or another traverser node that is still waiting for its other child nodes to return
    def _propagate_reward(self, node, reward):
        while True:
            parent = node.parent
            # If parent is none, the current node is a root node == traversal finished
            if parent is None:
                break
            # If required, calculate the expected value for the parent node, or directly propagate the reward one step further
            was_action_dont_care = node.action_type is None
            if parent.acting_player == self.traverser and not was_action_dont_care:
                if node.bet_bucket is not None:
                    parent.rewards_bet[node.bet_bucket] = reward
                    parent.n_rewards_left_bet -= 1
                    if parent.n_rewards_left_bet == 0:
                        # Calculate expected reward for the bet action
                        parent.rewards_action[2] = (parent.pi_bet * parent.rewards_bet).sum()
                        parent.n_rewards_left_action -= 1
                else:
                    parent.rewards_action[int(node.action_type)] = reward
                    parent.n_rewards_left_action -= 1
                # Once all available actions have a sampled reward, we calculate the expected reward, and the regrets for each action
                if parent.n_rewards_left_action == 0:
                    # Calculate the regrets (and set invalid action/bet regrets to 0)
                    expected_reward = (parent.pi_action * parent.rewards_action).sum()
                    action_regrets = parent.rewards_action - expected_reward
                    action_regrets[np.argwhere(parent.valid_actions == 0)] = 0
                    if parent.valid_actions[2]:
                        bet_expected_reward = parent.rewards_action[2]
                        bet_regrets = parent.rewards_bet - bet_expected_reward
                        bet_regrets[parent.invalid_bets] = 0
                    else:
                        bet_regrets = np.zeros(N_BET_BUCKETS, dtype=np.float32)
                    self.regret_obs_list.append(parent.observation_history)
                    self.regret_obs_count_list.append(parent.observation_count)
                    self.regret_action_list.append(action_regrets)
                    self.regret_bet_list.append(bet_regrets)
                    self._propagate_reward(parent, expected_reward)
                    self.all_nodes.pop(node.id)
                break
            else:
                self.all_nodes.pop(node.id)
                node = parent

    # Creates a single child node for the traverser, and step through it with the specified action/bet
    def _create_child_node(self, parent, action, bet, observations, observation_counts, mappings, prev_obs):
        env = parent.env
        env_copy = copy.deepcopy(env)
        self.all_nodes[self.node_n] = Node(self.node_n, parent, env_copy, action, bet)
        node = self.all_nodes[self.node_n]
        obs, obs_count, reward, done = node.step(prev_obs, parent_is_traverser=True)
        if not done:
            self.waiting_nodes.add(self.node_n)
            observations[node.acting_player].append(obs)
            observation_counts[node.acting_player].append(obs_count)
            mappings[self.node_n] = [node.acting_player, len(observations[node.acting_player]) - 1]
        else:
            self._propagate_reward(node, reward)
        self.node_n += 1

    # Create all available child nodes for the traverser (one node for each available action and bet size), and step through them
    def _create_child_nodes(self, obs, parent, observations, observation_counts, mappings):
        valid_actions = obs[indices.VALID_ACTIONS]
        min_bet = obs[indices.VALID_BET_LOW]
        max_bet = obs[indices.VALID_BET_HIGH]
        bet_sizes = np.concatenate(
            [np.array([min_bet, (min_bet + max_bet) / 2, max_bet]), BET_BUCKETS * obs[indices.POT_SIZE]]
        )
        valid_bets = np.logical_and(min_bet <= bet_sizes, bet_sizes <= max_bet)
        parent.n_rewards_left_action = len(np.argwhere(valid_actions == 1))
        parent.n_rewards_left_bet = len(np.argwhere(valid_bets))
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
        # Iterate over all waiting nodes and:
        #   1) add predicted regrets to them
        #   2) sample child nodes with external sampling
        for key in list(self.waiting_nodes):
            self.waiting_nodes.remove(key)
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
                    del node.env  # Save some memory by removing (now) unused environment
                    continue
                else:
                    parent_is_traverser = True
            env = node.env
            env_copy = copy.deepcopy(env)
            del node.env  # Save some memory by removing (now) unused environment
            self.all_nodes[self.node_n] = Node(self.node_n, node, env_copy)
            new_node = self.all_nodes[self.node_n]
            obs, obs_count, reward, done = new_node.step(previous_observation, parent_is_traverser, action_dont_care)
            if not parent_is_traverser:
                self.strategy_obs_list[node.acting_player].append(node.observation_history)
                self.strategy_obs_count_list[node.acting_player].append(node.observation_count)
                self.strategy_action_list[node.acting_player].append(node.pi_action)
                self.strategy_bet_list[node.acting_player].append(node.pi_bet)
            if not done:
                self.waiting_nodes.add(self.node_n)
                observations[new_node.acting_player].append(obs)
                observation_counts[new_node.acting_player].append(obs_count)
                new_mappings[self.node_n] = [new_node.acting_player, len(observations[new_node.acting_player]) - 1]
            else:
                self._propagate_reward(new_node, reward)
            self.node_n += 1

        self.regret_que.put((self.regret_obs_list, self.regret_obs_count_list, self.regret_action_list, self.regret_bet_list))
        for player in range(N_PLAYERS):
            if player != self.traverser:
                self.strategy_ques[player].put((self.strategy_obs_list[player], self.strategy_obs_count_list[player], self.strategy_action_list[player], self.strategy_bet_list[player]))
        self.regret_obs_list = []
        self.regret_obs_count_list = []
        self.regret_action_list = []
        self.regret_bet_list = []
        self.strategy_obs_list = [[] for _ in range(N_PLAYERS)]
        self.strategy_obs_count_list = [[] for _ in range(N_PLAYERS)]
        self.strategy_action_list = [[] for _ in range(N_PLAYERS)]
        self.strategy_bet_list = [[] for _ in range(N_PLAYERS)]

        gc.collect()

        return observations, observation_counts, new_mappings


# Represents a single decision point in a trajectory traversal
class Node:
    def __init__(self, id, parent, env, action_type=None, bet_bucket=None):
        self.id = id
        self.parent = parent
        self.env = env
        self.acting_player = None
        # The observation history, under which the action leading to the next node is/was taken
        self.observation_history = np.zeros([SEQUENCE_LENGTH, OBS_SHAPE], dtype=np.float32)
        self.observation_count = 0
        # Which action was applied to the PARENT environment to reach this node (used to fill the parent reward array)
        self.action_type = action_type
        # Which bet was applied to the PARENT environment to reach this node (used to fill the parent reward array)
        self.bet_bucket = bet_bucket
        # Rewards from child nodes (for each action/bet size taken from this node)
        self.rewards_action = np.zeros(N_ACTIONS, dtype=np.float32)
        self.rewards_bet = np.zeros(N_BET_BUCKETS, dtype=np.float32)
        # How many child rewards have been added (used to track when to calculate expected reward/regrets)
        self.n_rewards_left_action = np.zeros(N_ACTIONS)
        self.n_rewards_left_bet = np.zeros(N_BET_BUCKETS)
        # Behavior policies for acting and betting, calculated from predicted regrets
        self.pi_action = np.zeros(N_ACTIONS, dtype=np.float32)
        self.pi_bet = np.zeros(N_BET_BUCKETS, dtype=np.float32)
        self.valid_actions = None
        self.invalid_bets = None

    def reset(self):
        obs = self.env.reset()
        self.acting_player = int(obs[indices.ACTING_PLAYER])
        self.observation_history[0] = obs
        self.observation_count += 1
        return self.observation_history

    def add_regrets(self, previous_obs, action_regrets, bet_regrets):
        action_regrets = action_regrets.copy()
        bet_regrets = bet_regrets.copy()
        self.valid_actions = previous_obs[indices.VALID_ACTIONS]
        min_bet = previous_obs[indices.VALID_BET_LOW]
        max_bet = previous_obs[indices.VALID_BET_HIGH]
        bet_sizes = np.concatenate(
            [np.array([min_bet, (min_bet + max_bet) / 2, max_bet]), BET_BUCKETS * previous_obs[indices.POT_SIZE]])
        self.invalid_bets = np.logical_or(min_bet > bet_sizes, bet_sizes > max_bet)

        # When all regrets are negative, select the least negative entry deterministically like in the DCRM paper
        if np.count_nonzero(action_regrets >= 0) == 0:
            chosen_i = action_regrets.argmax()
            action_regrets[chosen_i] = 1
        action_regrets = np.maximum(action_regrets, np.zeros_like(action_regrets), dtype=np.float32)
        # Make the calculated policy probabilities sum to 1
        if np.count_nonzero(action_regrets) == 0:
            # If all regrets are zero, give all (valid) actions uniform probability
            self.pi_action[np.argwhere(self.valid_actions == 1)] = 1 / np.count_nonzero(self.valid_actions)
        else:
            self.pi_action = action_regrets / action_regrets.sum()

        # If bet action is not available bet pi should be all zeros (so bet loss will evaluate to 0 when training strategy network)
        if self.valid_actions[2]:
            # When all regrets are negative, select the least negative entry deterministically like in the DCRM paper
            if np.count_nonzero(bet_regrets >= 0) == 0:
                chosen_i = bet_regrets.argmax()
                bet_regrets[chosen_i] = 1
            bet_regrets = np.maximum(bet_regrets, np.zeros_like(bet_regrets), dtype=np.float32)
            # Make the calculated policy probabilities sum to 1
            if np.count_nonzero(bet_regrets) == 0:
                # If all regrets are zero, give all (valid) bets uniform probability
                self.pi_bet[np.logical_not(self.invalid_bets)] = 1 / np.count_nonzero(np.logical_not(self.invalid_bets))
            else:
                self.pi_bet = bet_regrets / bet_regrets.sum()

    # Steps the node environment (copy of parent env) forward by a single time step, with the action decided by parent node
    def step(self, previous_obs, parent_is_traverser=False, action_dont_care=False):
        table_action = Action(PlayerAction.CHECK, 0)
        # If the action is a "don't care", we are only feeding junk actions to the environment in order to get the final rewards back,
        # otherwise, apply the action/bet provided by the parent (if traverser), or sample them from the (parent) behaviour policies
        if not action_dont_care:
            min_bet = previous_obs[indices.VALID_BET_LOW]
            max_bet = previous_obs[indices.VALID_BET_HIGH]
            bet_sizes = np.concatenate(
                [np.array([min_bet, (min_bet + max_bet) / 2, max_bet]), BET_BUCKETS * previous_obs[indices.POT_SIZE]])
            # If action type is None, then the acting player is not the traverser, and we sample the action from the (parent) distribution
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
        # If we are not done (have not received final reward for traverser),
        # then we need to construct an up to date observation history for the next player to act
        if not done:
            self.observation_history = np.zeros([SEQUENCE_LENGTH, OBS_SHAPE], dtype=np.float32)
            self.observation_history[0] = obs
            self.observation_count = 1
            next_player = int(obs[indices.ACTING_PLAYER])
            self.acting_player = next_player
            p_parent = self.parent
            # Find the most recent decision node where the current acting player was the acting player, and copy the observation history
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
