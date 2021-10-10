from pokerenv.table import Table
from pokerenv.common import action_list, PlayerAction, Action
from copy import deepcopy
from config import BET_BUCKETS, SEQUENCE_LENGTH, OBS_SHAPE, N_PLAYERS, N_ACTIONS, N_BET_BUCKETS, PB_C_BASE, PB_C_INIT, INITIAL_VALUE, \
    DIRICHLET_ALPHA, EXPLORATION_FRACTION, GLOBAL_EVAL_HOST, PLAYER_ACTOR_HOST_MAP, N_MONTE_CARLO_SIMS, EVAL_CONSIDER_SINGLE_TRAJECTORY, \
    MASTER_HOST, N_LOGGED_HANDS, N_MCTS_EVAL_HANDS, STUPID_MCTS
from random import shuffle as rshuffle
from rpc.RL_pb2_grpc import ActorStub, EvalMCTSStub, MasterStub
from rpc.RL_pb2 import Observation, SampledEvalData, Empty, FloatMessage
from utils import fast_deep_copy
from operator import itemgetter
import pokerenv.obs_indices as indices
import math
import numpy as np
import grpc
import os
import logging

simulated_player = 0


def run_evaluations(eval_iteration):
    options = [('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)]
    master_channel = grpc.insecure_channel(MASTER_HOST, options)
    master_stub = MasterStub(master_channel)
    strategy_stubs = {player: ActorStub(grpc.insecure_channel(PLAYER_ACTOR_HOST_MAP[player], options)) for player in range(1, N_PLAYERS)}
    eval_channel = grpc.insecure_channel(GLOBAL_EVAL_HOST, options)
    eval_stub = EvalMCTSStub(eval_channel)
    if 'iteration_%d' % eval_iteration not in os.listdir('mcts_hands'):
        os.makedirs('mcts_hands/iteration_%d/' % eval_iteration)
    table = Table(N_PLAYERS, player_names=['best_response', 'DCRM policy'], hand_history_location='mcts_hands/iteration_%d/' % eval_iteration)
    agents = {i: PolicyAgent() for i in range(1, N_PLAYERS)}
    agents[simulated_player] = MonteCarloAgent()
    while True:
        response = master_stub.RequestMCTSEvaluation(Empty())
        if response.value == -1:
            master_stub.ExitMCTSEvaluationPool(Empty())
            break
        if np.random.uniform() > 1 - (N_LOGGED_HANDS / N_MCTS_EVAL_HANDS):
            table.hand_history_enabled = True
        else:
            table.hand_history_enabled = False
        obs = table.reset()
        rewards = np.zeros(N_PLAYERS)
        while True:
            acting_player = int(obs[indices.ACTING_PLAYER])
            if acting_player != simulated_player:
                table_action = agents[acting_player].get_action(strategy_stubs[acting_player], obs)
            else:
                other_observations = [a.observations for k, a in agents.items()]
                other_observation_counts = [a.obs_count for k, a in agents.items()]
                table_action = agents[acting_player].get_action(table, obs, other_observations, other_observation_counts, strategy_stubs, eval_stub)
            obs, reward, env_done, _ = table.step(table_action)
            if obs[indices.HAND_IS_OVER] and acting_player != simulated_player:
                rewards[acting_player] = reward
            if env_done:
                for key, agent in agents.items():
                    agent.reset()
                master_stub.AddMCTSExploitabilitySample(FloatMessage(value=-rewards.mean()))
                break


def construct_obs_proto(acting_player, observations, observation_counts):
    observations_bytes = np.ndarray.tobytes(np.asarray(observations, dtype=np.float32))
    player_obs_count_bytes = np.ndarray.tobytes(np.expand_dims(np.asarray(observation_counts, dtype=np.int32), 0))
    obs_proto = Observation(player=acting_player, observations=observations_bytes, observation_counts=player_obs_count_bytes,
                            shape=1, sequence_length=SEQUENCE_LENGTH)
    return obs_proto


def get_bet_sizes(obs):
    min_bet = obs[indices.VALID_BET_LOW]
    max_bet = obs[indices.VALID_BET_HIGH]
    return np.concatenate([np.array([min_bet, (min_bet + max_bet) / 2, max_bet]), BET_BUCKETS * obs[indices.POT_SIZE]])


def update_obs_sequence(observations, obs, obs_count):
    obs_count = min(obs_count + 1, SEQUENCE_LENGTH)
    if obs_count == SEQUENCE_LENGTH:
        observations[0:SEQUENCE_LENGTH - 1] = observations[1:SEQUENCE_LENGTH]
        observations[-1] = obs
    else:
        observations[obs_count - 1] = obs
    return obs_count


class Agent:
    def __init__(self, initial_obs=None, initial_obs_count=None):
        if initial_obs is None:
            self.observations = np.zeros([SEQUENCE_LENGTH, OBS_SHAPE])
            self.obs_count = 0
        else:
            self.observations = initial_obs
            self.obs_count = initial_obs_count

    def reset(self, initial_obs=None, initial_obs_count=None):
        if initial_obs is None:
            self.observations = np.zeros([SEQUENCE_LENGTH, OBS_SHAPE])
            self.obs_count = 0
        else:
            self.observations = initial_obs
            self.obs_count = initial_obs_count

    def append_obs(self, obs):
        new_obs_count = update_obs_sequence(self.observations, obs, self.obs_count)
        self.obs_count = new_obs_count


class PolicyAgent(Agent):
    def __init__(self, initial_obs=None, initial_obs_count=None):
        super(PolicyAgent, self).__init__(initial_obs, initial_obs_count)

    def get_action(self, strategy_stub, obs):
        if not obs[indices.HAND_IS_OVER]:
            self.obs_count = min(self.obs_count + 1, SEQUENCE_LENGTH)
            self.append_obs(obs)
            bet_sizes = get_bet_sizes(obs)
            observations_bytes = np.ndarray.tobytes(np.asarray(self.observations, dtype=np.float32))
            player_obs_count_bytes = np.ndarray.tobytes(np.expand_dims(np.asarray(self.obs_count, dtype=np.int32), 0))
            obs_proto = Observation(player=int(obs[indices.ACTING_PLAYER]), observations=observations_bytes,
                                    observation_counts=player_obs_count_bytes, shape=1, sequence_length=SEQUENCE_LENGTH)
            response = strategy_stub.GetStrategies(obs_proto)
            action_pred = np.frombuffer(response.action_prediction, dtype=np.float32).reshape(N_ACTIONS)
            bet_pred = np.frombuffer(response.bet_prediction, dtype=np.float32).reshape(N_BET_BUCKETS)
            action = np.random.choice(action_list, p=np.exp(action_pred))
            action = action_list[action]
            bet_size = 0
            if action == PlayerAction.BET:
                bet_size = np.random.choice(bet_sizes, p=np.exp(bet_pred))
            table_action = Action(action, bet_size)
        else:
            table_action = Action(PlayerAction.CHECK, 0)
        return table_action


class MonteCarloAgent(Agent):
    def __init__(self, initial_obs=None, initial_obs_count=None):
        super(MonteCarloAgent, self).__init__(initial_obs, initial_obs_count)

    def get_action(self, table, obs, others_observation_sequences, others_observation_counts, strategy_stubs, eval_stub):
        if not obs[indices.HAND_IS_OVER]:
            self.append_obs(obs)
            all_observation_sequences = [self.observations.copy()] + others_observation_sequences
            all_observation_counts = [self.obs_count] + others_observation_counts
            table_action = run_mcts_simulations(N_MONTE_CARLO_SIMS, table, obs, all_observation_sequences, all_observation_counts, strategy_stubs, eval_stub)
        else:
            table_action = Action(PlayerAction.CHECK, 0)
        return table_action


def send_observed_trajectory(evaluator_stub, observations, obs_counts, a_p, b_p, reward):
    shape = len(observations)
    observations_bytes = np.ndarray.tobytes(np.asarray(observations, dtype=np.float32))
    player_obs_count_bytes = np.ndarray.tobytes(np.expand_dims(np.asarray(obs_counts, dtype=np.int32), 0))
    action_prior_bytes = np.ndarray.tobytes(np.asarray(a_p, dtype=np.float32))
    bet_prior_bytes = np.ndarray.tobytes(np.asarray(b_p, dtype=np.float32))
    value_bytes = np.ndarray.tobytes(np.asarray(reward, dtype=np.float32))
    obs_proto = SampledEvalData(player=0, observations=observations_bytes, observation_counts=player_obs_count_bytes,
                                action_prior=action_prior_bytes, bet_prior=bet_prior_bytes, values=value_bytes,
                                shape=shape, sequence_length=SEQUENCE_LENGTH)
    _ = evaluator_stub.AddValues(obs_proto)


def get_state_value_p_prediction(eval_stub, obs_proto):
    response = eval_stub.GetValues(obs_proto)
    value = np.frombuffer(response.value_prediction, dtype=np.float32).reshape(1)[0]
    action_prior = np.frombuffer(response.action_prior_prediction, dtype=np.float32).reshape(N_ACTIONS)
    bet_prior = np.frombuffer(response.bet_prior_prediction, dtype=np.float32).reshape(N_BET_BUCKETS)
    return value, action_prior, bet_prior


def uniform_expand():
    pass


def run_mcts_simulations(n_simulations, table, current_obs, all_observation_sequences, all_obs_counts, strategy_stubs, eval_stub):
    root = Node(0, 0)
    for n in range(n_simulations):
        obs = current_obs.copy()
        action_dont_care = obs[indices.HAND_IS_OVER]
        table_clone = fast_deep_copy(table)#deepcopy(table)
        table_clone.hand_history_enabled = False
        if not EVAL_CONSIDER_SINGLE_TRAJECTORY:
            # Shuffle the remaining deck cards to avoid drawing the same exact table cards each time
            rshuffle(table_clone.deck.cards)
        agents = {i: PolicyAgent(all_observation_sequences[i].copy(), all_obs_counts[i]) for i in range(1, N_PLAYERS)}
        inference_observation_sequence = all_observation_sequences[0].copy()
        inference_observation_count = all_obs_counts[0]
        node = root
        search_path = []
        traversed_obs_sequence_history = [inference_observation_sequence.copy()]
        traversed_obs_count_history = [inference_observation_count]
        acting_player = 0
        final_reward = None
        while True:
            bet_sizes = get_bet_sizes(obs)
            if not action_dont_care:
                if acting_player != simulated_player:
                    table_action = agents[acting_player].get_action(strategy_stubs[acting_player], obs)
                else:
                    # If the node has not been expanded (traversed through) yet:
                    if node.children is None:
                        action_prior = np.full(N_ACTIONS, 1/N_ACTIONS)
                        bet_prior = np.full(N_BET_BUCKETS, 1/N_BET_BUCKETS)
                        if not STUPID_MCTS:
                            # Update the node average value and action/bet priori based on a value_p network prediction:
                            obs_proto = construct_obs_proto(acting_player, inference_observation_sequence, inference_observation_count)
                            value, action_prior, bet_prior = get_state_value_p_prediction(eval_stub, obs_proto)
                        # Create all the child nodes (all possible actions from this node):
                        node.children = {}
                        node.expand_children(action_prior, bet_prior)
                        if node == root:
                            add_exploration_noise(node)
                        if not STUPID_MCTS:
                            break
                    search_path.append(node)
                    action, node = select_child(node, obs)
                    table_action = Action(action_list[action[0]], bet_sizes[action[1]])
            else:
                table_action = Action(PlayerAction.CHECK, 0)

            obs, reward, _, _ = table_clone.step(table_action)
            done = obs[indices.HAND_IS_OVER] and acting_player == simulated_player
            # We are done once the simulated player receives a final reward
            if done:
                if not STUPID_MCTS:
                    # Even after receiving a sampled reward, we still update the node value based on the value_p network prediction
                    obs_proto = construct_obs_proto(acting_player, inference_observation_sequence, inference_observation_count)
                    value, _, _ = get_state_value_p_prediction(eval_stub, obs_proto)
                    # Save the final reward to train the value_p network
                    final_reward = reward
                else:
                    value = reward
                break
            else:
                acting_player = int(obs[indices.ACTING_PLAYER])
                action_dont_care = obs[indices.HAND_IS_OVER]
                if acting_player == simulated_player and not action_dont_care:
                    # For valid action points, update the simulated player inference sequence and trajectory
                    new_obs_count = update_obs_sequence(inference_observation_sequence, obs, inference_observation_count)
                    inference_observation_count = new_obs_count
                    traversed_obs_sequence_history.append(inference_observation_sequence.copy())
                    traversed_obs_count_history.append(inference_observation_count)
        # After the simulation is done (received final reward or reached a leaf node),
        # update the values of nodes we traversed through, using a value_p network estimate
        for node in search_path:
            node.visit_count += 1
            node.value_sum += value
        if not STUPID_MCTS:
            # If we received a final reward, send data to update the value_p network, with the sampled final reward and action/bet probability
            if len(search_path) > 0 and final_reward is not None:
                observations = np.asarray(traversed_obs_sequence_history)
                observation_counts = np.asarray(traversed_obs_count_history)
                a_p, b_p = get_normalized_visit_counts(search_path)
                send_observed_trajectory(eval_stub, observations, observation_counts, a_p, b_p, final_reward*np.ones_like(observation_counts))
    return select_action(root, current_obs)


class Node:
    def __init__(self, action_prior, bet_prior):
        self.action_prior = action_prior
        self.bet_prior = bet_prior
        self.value_sum = 0
        self.visit_count = 0
        self.children = None

    def value(self):
        if self.visit_count == 0:
            return INITIAL_VALUE
        else:
            return self.value_sum / self.visit_count

    def expand_children(self, action_prior, bet_prior):
        action_denom = action_prior.sum()
        bet_denom = bet_prior.sum()
        for a in range(N_ACTIONS):
            if a == 3:
                for b in range(N_BET_BUCKETS):
                    self.children[(a, b)] = Node(action_prior[a] / action_denom, bet_prior[b] / bet_denom)
            else:
                self.children[(a, 1)] = Node(action_prior[a] / action_denom, bet_prior[0] / bet_denom)


def get_normalized_visit_counts(search_path):
    actions = []
    bets = []
    for node in search_path:
        assert node.children is not None
        total = sum([child.visit_count for key, child in node.children.items()]) + 1
        normalized_a = [child.visit_count / total for key, child in node.children.items() if key[0] != 3]
        bet_prob = 1-sum(normalized_a)
        normalized_a.append(bet_prob)
        actions.append(normalized_a)
        total_bet = sum([child.visit_count for key, child in node.children.items() if key[0] == 3]) + 1
        normalized_b = [child.visit_count / total_bet for key, child in node.children.items() if key[0] == 3]
        bets.append(normalized_b)
    return actions, bets


def select_action(root, obs):
    valid_actions = obs[indices.VALID_ACTIONS]
    min_bet = obs[indices.VALID_BET_LOW]
    max_bet = obs[indices.VALID_BET_HIGH]
    bet_sizes = get_bet_sizes(obs)
    valid_bets = np.logical_and(min_bet <= bet_sizes, bet_sizes <= max_bet)
    visit_counts = [(child.visit_count, action)
                    for action, child in root.children.items() if valid_actions[action[0]] and valid_bets[action[1]]]
    rshuffle(visit_counts)
    _, action = max(visit_counts, key=itemgetter(0))
    return Action(action_list[action[0]], bet_sizes[action[1]])


def select_child(node, obs):
    valid_actions = obs[indices.VALID_ACTIONS]
    min_bet = obs[indices.VALID_BET_LOW]
    max_bet = obs[indices.VALID_BET_HIGH]
    bet_sizes = get_bet_sizes(obs)
    valid_bets = np.logical_and(min_bet <= bet_sizes, bet_sizes <= max_bet)
    ucb_scores = [(ucb_score(node, child), action, child) for action, child in node.children.items() if valid_actions[action[0]] and valid_bets[action[1]]]
    rshuffle(ucb_scores)
    _, action, child = max(ucb_scores, key=itemgetter(0))

    return action, child


def ucb_score(parent, child):
    pb_c = math.log((parent.visit_count + PB_C_BASE + 1) / PB_C_BASE) + PB_C_INIT
    pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)
    prior_score = pb_c * (child.action_prior*child.bet_prior)
    value_score = child.value()
    return prior_score + value_score


def add_exploration_noise(node):
    actions = node.children.keys()
    noise = np.random.gamma(DIRICHLET_ALPHA, 1, len(actions))
    frac = EXPLORATION_FRACTION
    for a, n in zip(actions, noise):
        node.children[a].action_prior = node.children[a].action_prior * (1 - frac) + n * frac
        node.children[a].bet_prior = node.children[a].bet_prior * (1 - frac) + n * frac
