from pokerenv.table import Table
from pokerenv.common import action_list, PlayerAction, Action
from copy import deepcopy
from config import BET_BUCKETS, SEQUENCE_LENGTH, OBS_SHAPE, N_PLAYERS, N_ACTIONS, N_BET_BUCKETS, PB_C_BASE, PB_C_INIT, INITIAL_VALUE, \
    DIRICHLET_ALPHA, EXPLORATION_FRACTION, GLOBAL_STRATEGY_HOST, GLOBAL_EVAL_HOST
from rpc.RL_pb2_grpc import ActorStub, EvaluatorStub
from rpc.RL_pb2 import Observation
import pokerenv.obs_indices as indices
import numpy as np
import math
import numpy
import grpc


class Evaluator:
    def __init__(self, n_evaluations):
        self.n_evaluations = n_evaluations

    def run_evaluations(self):
        options = [('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)]
        strategy_channel = grpc.insecure_channel(GLOBAL_STRATEGY_HOST)
        eval_channel = grpc.insecure_channel(GLOBAL_EVAL_HOST)
        table = Table(2)
        for n in range(self.n_evaluations):
            buffer = MemoryBuffer()
            obs = table.reset()

            if obs[indices.ACTING_PLAYER] == 0:
                min_bet = obs[indices.VALID_BET_LOW]
                max_bet = obs[indices.VALID_BET_HIGH]
                bet_sizes = np.concatenate([np.array([min_bet, (min_bet + max_bet) / 2, max_bet]), BET_BUCKETS * obs[indices.POT_SIZE]])
                obs_proto = construct_proto(acting_player, observations, observation_counts)
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
                table_action = run_mcts_simulations(buffer, strategy_channel, eval_channel, 8000, table, obs)

            pass


class MemoryBuffer:
    def __init__(self):
        pass

    def add_experience(self, obs, obs_count, reward):
        pass

    def send_to_server(self):
        pass


def construct_proto(acting_player, observations, observation_counts):
    observation_count = observation_counts[acting_player]
    player_observations = observations[acting_player]
    observations_bytes = np.ndarray.tobytes(np.asarray(player_observations))
    player_obs_count_bytes = np.ndarray.tobytes(np.expand_dims(np.asarray(observation_count, dtype=np.int32), 0))
    obs_proto = Observation(player=acting_player, observations=observations_bytes, observation_counts=player_obs_count_bytes,
                            shape=1, sequence_length=SEQUENCE_LENGTH)
    return obs_proto


def run_mcts_simulations(memory, strategy_channel, value_channel, n_simulations, table, initial_obs):
    strategy_stub = ActorStub(strategy_channel)
    value_stub = EvaluatorStub(value_channel)
    root = Node(0, 0)

    action_dont_care = initial_obs[indices.DELAYED_REWARD]
    if action_dont_care:
        return Action(PlayerAction.CHECK, 0)

    for n in range(n_simulations):
        obs = initial_obs.copy()
        table_clone = deepcopy(table)
        observations = [np.zeros(SEQUENCE_LENGTH, OBS_SHAPE) for _ in range(N_PLAYERS)]
        observation_counts = [0 for _ in range(N_PLAYERS)]
        node = root
        search_tree = [node]
        acting_player = obs[indices.ACTING_PLAYER]
        observations[acting_player][0] = obs
        observation_counts[acting_player] = 1
        tracked_obs_history = [observations[acting_player].copy()]
        tracked_obs_count = [observation_counts[acting_player]]
        final_reward = None
        while True:
            min_bet = obs[indices.VALID_BET_LOW]
            max_bet = obs[indices.VALID_BET_HIGH]
            bet_sizes = np.concatenate([np.array([min_bet, (min_bet + max_bet) / 2, max_bet]), BET_BUCKETS * obs[indices.POT_SIZE]])
            table_action = Action(PlayerAction.CHECK, 0)
            if not action_dont_care:
                if acting_player == 0:
                    obs_proto = construct_proto(acting_player, observations, observation_counts)
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
                    if node.children is None:
                        obs_proto = construct_proto(acting_player, observations, observation_counts)
                        response = value_stub.GetValues(obs_proto)
                        value = np.frombuffer(response.value_prediction, dtype=np.float32).reshape(1)[0]
                        action_prior = np.frombuffer(response.action_prior_prediction, dtype=np.float32).reshape(N_ACTIONS)
                        bet_prior = np.frombuffer(response.bet_prior_prediction, dtype=np.float32).reshape(N_BET_BUCKETS)
                        action_denom = action_prior.sum()
                        bet_denom = bet_prior.sum()
                        node.children = {}
                        for a in range(N_ACTIONS):
                            if a == 3:
                                for b in range(N_BET_BUCKETS):
                                    node.children[(a, b)]: Node(action_prior[a]/action_denom, bet_prior[b]/bet_denom)
                            else:
                                node.children[(a, 1)]: Node(action_prior[a]/action_denom, bet_prior[0]/bet_denom)
                        if node == root:
                            add_exploration_noise(node)
                        break
                    else:
                        action, node = select_child(node, obs)
                        search_tree.append(node)
                        table_action = Action(action_list[action[0]], bet_sizes[action[1]])
            obs, reward, env_done = table_clone.step(table_action)
            done = env_done or ((obs[indices.HAND_IS_OVER] or action_dont_care) and acting_player == 0)
            if done:
                obs_proto = construct_proto(acting_player, observations, observation_counts)
                response = value_stub.GetValues(obs_proto)
                value = np.frombuffer(response.value_prediction, dtype=np.float32).reshape(1)[0]
                final_reward = reward
                break
            else:
                acting_player = obs[indices.ACTING_PLAYER]
                action_dont_care = obs[indices.DELAYED_REWARD]
                if not action_dont_care:
                    observation_count = min(observation_counts[acting_player] + 1, SEQUENCE_LENGTH)
                    if observation_count == SEQUENCE_LENGTH:
                        observations[0:SEQUENCE_LENGTH - 1] = observations[1:SEQUENCE_LENGTH]
                        observations[-1] = obs
                    else:
                        observations[observation_count - 1] = obs
                    if acting_player != 0:
                        tracked_obs_history.append(observations[acting_player].copy())
                        tracked_obs_count.append(observation_count)
        for node in search_tree:
            node.visit_count += 1
            node.value_sum += value
        if final_reward is not None:
            obs = np.asarray(tracked_obs_history)
            obs_count = np.asarray(tracked_obs_count)
            memory.add_experience(obs, obs_count, final_reward*np.ones_like(tracked_obs_count))
    min_bet = initial_obs[indices.VALID_BET_LOW]
    max_bet = initial_obs[indices.VALID_BET_HIGH]
    bet_sizes = np.concatenate([np.array([min_bet, (min_bet + max_bet) / 2, max_bet]), BET_BUCKETS * initial_obs[indices.POT_SIZE]])
    chosen_action = select_action(root, initial_obs)
    return Action(action_list[chosen_action[0]], bet_sizes[chosen_action[1]])


def select_action(root, obs):
    valid_actions = obs[indices.VALID_ACTIONS]
    min_bet = obs[indices.VALID_BET_LOW]
    max_bet = obs[indices.VALID_BET_HIGH]
    bet_sizes = np.concatenate(
        [np.array([min_bet, (min_bet + max_bet) / 2, max_bet]), BET_BUCKETS * obs[indices.POT_SIZE]]
    )
    valid_bets = np.logical_and(min_bet <= bet_sizes, bet_sizes <= max_bet)
    visit_counts = [(child.visit_count, action)
                    for action, child in root.children.items() if valid_actions[action[0]] and valid_bets[action[1]]]
    _, action = max(visit_counts)
    return action


def select_child(node, obs):
    valid_actions = obs[indices.VALID_ACTIONS]
    min_bet = obs[indices.VALID_BET_LOW]
    max_bet = obs[indices.VALID_BET_HIGH]
    bet_sizes = np.concatenate(
        [np.array([min_bet, (min_bet + max_bet) / 2, max_bet]), BET_BUCKETS * obs[indices.POT_SIZE]]
    )
    valid_bets = np.logical_and(min_bet <= bet_sizes, bet_sizes <= max_bet)
    _, action, child = max((ucb_score(node, child), action, child) for action, child in node.children.items()
                           if valid_actions[action[0]] and valid_bets[action[1]])
    return action, child


def ucb_score(parent, child):
    pb_c = math.log((parent.visit_count + PB_C_BASE + 1) /
                    PB_C_BASE) + PB_C_INIT
    pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)
    prior_score = pb_c * (child.action_prior*child.bet_prior)
    value_score = child.value()
    return prior_score + value_score


def add_exploration_noise(node):
  actions = node.children.keys()
  noise = numpy.random.gamma(DIRICHLET_ALPHA, 1, len(actions))
  frac = EXPLORATION_FRACTION
  for a, n in zip(actions, noise):
    node.children[a].action_prior = node.children[a].action_prior * (1 - frac) + n * frac
    node.children[a].bet_prior = node.children[a].bet_prior * (1 - frac) + n * frac


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
