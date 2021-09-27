from pokerenv.table import Table
from pokerenv.common import action_list, PlayerAction, Action
from copy import deepcopy
from config import BET_BUCKETS, SEQUENCE_LENGTH, OBS_SHAPE, N_PLAYERS, N_ACTIONS, N_BET_BUCKETS, PB_C_BASE, PB_C_INIT, INITIAL_VALUE, \
    DIRICHLET_ALPHA, EXPLORATION_FRACTION, GLOBAL_STRATEGY_HOST, GLOBAL_EVAL_HOST, NUM_EVAL_TRAINING_LOOPS, PLAYER_ACTOR_HOST_MAP, N_MONTE_CARLO_SIMS, HH_LOCATION
from rpc.RL_pb2_grpc import ActorStub, EvaluatorStub
from rpc.RL_pb2 import Observation, SampledEvalData, Empty
import pokerenv.obs_indices as indices
import math
import numpy as np
import grpc
import logging


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


class StrategyAgent(Agent):
    def __init__(self, channel, initial_obs=None, initial_obs_count=None):
        super(StrategyAgent, self).__init__(initial_obs, initial_obs_count)
        self.strategy_stub = ActorStub(channel)

    def get_action(self, obs):
        if not obs[indices.HAND_IS_OVER]:
            self.obs_count = min(self.obs_count + 1, SEQUENCE_LENGTH)

            if self.obs_count == SEQUENCE_LENGTH:
                self.observations[0:SEQUENCE_LENGTH - 1] = self.observations[1:SEQUENCE_LENGTH]
                self.observations[-1] = obs
            else:
                self.observations[self.obs_count - 1] = obs

            min_bet = obs[indices.VALID_BET_LOW]
            max_bet = obs[indices.VALID_BET_HIGH]
            bet_sizes = np.concatenate([np.array([min_bet, (min_bet + max_bet) / 2, max_bet]), BET_BUCKETS * obs[indices.POT_SIZE]])
            observations_bytes = np.ndarray.tobytes(np.asarray(self.observations, dtype=np.float32))
            player_obs_count_bytes = np.ndarray.tobytes(np.expand_dims(np.asarray(self.obs_count, dtype=np.int32), 0))
            obs_proto = Observation(player=int(obs[indices.ACTING_PLAYER]), observations=observations_bytes, observation_counts=player_obs_count_bytes,
                                    shape=1, sequence_length=SEQUENCE_LENGTH)
            response = self.strategy_stub.GetStrategies(obs_proto)
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
    def __init__(self, strategy_channel, value_channel, initial_obs=None, initial_obs_count=None):
        super(MonteCarloAgent, self).__init__(initial_obs, initial_obs_count)
        self.strategy_channel = strategy_channel
        self.value_channel = value_channel
        self.buffer = MemoryBuffer(value_channel)

    def get_action(self, table, obs, obs_histories, obs_history_counts):
        if not obs[indices.HAND_IS_OVER]:
            self.obs_count = min(self.obs_count + 1, SEQUENCE_LENGTH)
            if self.obs_count == SEQUENCE_LENGTH:
                self.observations[0:SEQUENCE_LENGTH - 1] = self.observations[1:SEQUENCE_LENGTH]
                self.observations[-1] = obs
            else:
                self.observations[self.obs_count - 1] = obs

            obs_histories = [self.observations.copy()] + obs_histories
            obs_history_counts = [self.obs_count] + obs_history_counts
            table_action = run_mcts_simulations(self.buffer, self.strategy_channel, self.value_channel, N_MONTE_CARLO_SIMS, table, obs, obs_histories, obs_history_counts)
        else:
            table_action = Action(PlayerAction.CHECK, 0)
        return table_action


def run_evaluations(n_evaluations):
    options = [('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)]
    strategy_channel = grpc.insecure_channel(PLAYER_ACTOR_HOST_MAP[0], options)
    value_channel = grpc.insecure_channel(GLOBAL_EVAL_HOST, options)
    value_stub = EvaluatorStub(value_channel)
    table = Table(N_PLAYERS, hand_history_location=HH_LOCATION)
    table.hand_history_enabled = True
    agents = {i: StrategyAgent(strategy_channel) for i in range(1, N_PLAYERS)}
    agents[0] = MonteCarloAgent(strategy_channel, value_channel)
    rewards = []
    hands = 0
    for n in range(n_evaluations):
        logging.info("Evaluation n: %d" % n)
        obs = table.reset()
        while True:
            acting_player = int(obs[indices.ACTING_PLAYER])
            action_dont_care = obs[indices.HAND_IS_OVER]
            if acting_player != 0:
                table_action = agents[acting_player].get_action(obs)
            else:
                other_observations = [a.observations for k, a in agents.items()]
                other_observation_counts = [a.obs_count for k, a in agents.items()]
                table_action = agents[acting_player].get_action(table, obs, other_observations, other_observation_counts)
                if n < NUM_EVAL_TRAINING_LOOPS:
                    agents[acting_player].buffer.send_to_server()
                    if agents[acting_player].buffer.has_data:
                        value_stub.TrainValues(Empty())
            obs, reward, env_done, _ = table.step(table_action)
            done = env_done or ((obs[indices.HAND_IS_OVER] or action_dont_care) and acting_player == 0)
            if done:
                if n > NUM_EVAL_TRAINING_LOOPS:
                    hands += 1
                    rewards.append(reward)
                for key, agent in agents.items():
                    agent.reset()
                break
    return 100 * sum(rewards) / hands


class MemoryBuffer:
    def __init__(self, channel):
        self.has_data = False
        self.evaluator_stub = EvaluatorStub(channel)
        self.observations = []
        self.obs_count = []
        self.reward = []
        self.a_p = []
        self.b_p = []

    def add_experience(self, obs, obs_count, reward, a_p, b_p):
        self.has_data = True
        self.observations.extend(obs)
        self.obs_count.extend(obs_count)
        self.reward.extend(reward)
        self.a_p.extend(a_p)
        self.b_p.extend(b_p)

    def send_to_server(self):
        shape = len(self.observations)
        observations_bytes = np.ndarray.tobytes(np.asarray(self.observations, dtype=np.float32))
        player_obs_count_bytes = np.ndarray.tobytes(np.expand_dims(np.asarray(self.obs_count, dtype=np.int32), 0))
        action_prior_bytes = np.ndarray.tobytes(np.asarray(self.a_p, dtype=np.float32))
        bet_prior_bytes = np.ndarray.tobytes(np.asarray(self.b_p, dtype=np.float32))
        value_bytes = np.ndarray.tobytes(np.asarray(self.reward, dtype=np.float32))
        obs_proto = SampledEvalData(player=0, observations=observations_bytes, observation_counts=player_obs_count_bytes,
                                action_prior=action_prior_bytes, bet_prior=bet_prior_bytes, values=value_bytes,
                                shape=shape, sequence_length=SEQUENCE_LENGTH)
        _ = self.evaluator_stub.AddValues(obs_proto)
        self.has_data = False
        self.observations = []
        self.obs_count = []
        self.reward = []
        self.a_p = []
        self.b_p = []


def construct_proto(acting_player, observations, observation_counts):
    observations_bytes = np.ndarray.tobytes(np.asarray(observations, dtype=np.float32))
    player_obs_count_bytes = np.ndarray.tobytes(np.expand_dims(np.asarray(observation_counts, dtype=np.int32), 0))
    obs_proto = Observation(player=acting_player, observations=observations_bytes, observation_counts=player_obs_count_bytes,
                            shape=1, sequence_length=SEQUENCE_LENGTH)
    return obs_proto


def run_mcts_simulations(memory, strategy_channel, value_channel, n_simulations, table, starting_obs, starting_obs_histories, starting_obs_counts):
    value_stub = EvaluatorStub(value_channel)
    root = Node(0, 0)

    for n in range(n_simulations):
        if n % 50 == 0:
            logging.info("monte carlo simulation n: %d" % n)
        obs = starting_obs.copy()
        action_dont_care = obs[indices.HAND_IS_OVER]
        table_clone = deepcopy(table)
        table_clone.hand_history_enabled = False
        agents = {i: StrategyAgent(strategy_channel, starting_obs_histories[i].copy(), starting_obs_counts[i]) for i in range(1, N_PLAYERS)}
        observations = starting_obs_histories[0].copy()
        observation_count = starting_obs_counts[0]
        node = root
        search_path = []
        tracked_obs_history = [observations.copy()]
        tracked_obs_count = [observation_count]

        acting_player = 0
        final_reward = None
        while True:
            min_bet = obs[indices.VALID_BET_LOW]
            max_bet = obs[indices.VALID_BET_HIGH]
            bet_sizes = np.concatenate([np.array([min_bet, (min_bet + max_bet) / 2, max_bet]), BET_BUCKETS * obs[indices.POT_SIZE]])
            table_action = Action(PlayerAction.CHECK, 0)
            if not action_dont_care:
                if acting_player != 0:
                    table_action = agents[acting_player].get_action(obs)
                else:
                    if node.children is None:
                        obs_proto = construct_proto(acting_player, observations, observation_count)
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
                                    node.children[(a, b)] = Node(action_prior[a]/action_denom, bet_prior[b]/bet_denom)
                            else:
                                node.children[(a, 1)] = Node(action_prior[a]/action_denom, bet_prior[0]/bet_denom)
                        if node == root:
                            add_exploration_noise(node)
                        break
                    else:
                        search_path.append(node)
                        action, node = select_child(node, obs)
                        table_action = Action(action_list[action[0]], bet_sizes[action[1]])
            obs, reward, env_done, _ = table_clone.step(table_action)
            done = env_done or ((obs[indices.HAND_IS_OVER] or action_dont_care) and acting_player == 0)
            if done:
                obs_proto = construct_proto(acting_player, observations, observation_count)
                response = value_stub.GetValues(obs_proto)
                value = np.frombuffer(response.value_prediction, dtype=np.float32).reshape(1)[0]
                final_reward = reward
                break
            else:
                acting_player = int(obs[indices.ACTING_PLAYER])
                action_dont_care = obs[indices.HAND_IS_OVER]
                if acting_player == 0 and not action_dont_care:
                    observation_count = min(observation_count + 1, SEQUENCE_LENGTH)
                    if observation_count == SEQUENCE_LENGTH:
                        observations[0:SEQUENCE_LENGTH - 1] = observations[1:SEQUENCE_LENGTH]
                        observations[-1] = obs
                    else:
                        observations[observation_count - 1] = obs
                    tracked_obs_history.append(observations.copy())
                    tracked_obs_count.append(observation_count)
        for node in search_path:
            node.visit_count += 1
            node.value_sum += value
        if final_reward is not None:
            obs = np.asarray(tracked_obs_history)
            obs_count = np.asarray(tracked_obs_count)
            a_p, b_p = get_normalized_visit_counts(search_path)
            memory.add_experience(obs, obs_count, final_reward*np.ones_like(tracked_obs_count), a_p, b_p)
    min_bet = starting_obs[indices.VALID_BET_LOW]
    max_bet = starting_obs[indices.VALID_BET_HIGH]
    bet_sizes = np.concatenate([np.array([min_bet, (min_bet + max_bet) / 2, max_bet]), BET_BUCKETS * starting_obs[indices.POT_SIZE]])
    chosen_action = select_action(root, starting_obs)
    return Action(action_list[chosen_action[0]], bet_sizes[chosen_action[1]])


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
    return np.asarray(actions), np.asarray(bets)


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
  noise = np.random.gamma(DIRICHLET_ALPHA, 1, len(actions))
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
