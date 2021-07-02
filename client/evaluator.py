import numpy as np
import grpc
from rpc import RL_pb2_grpc
from rpc.RL_pb2 import Observation, Prediction, Selection
from pokerenv.common import PlayerAction, Action, action_list
from pokerenv.table import Table
from config import N_ACTIONS, N_BET_BUCKETS, SEQUENCE_LENGTH, BET_BUCKETS
import pokerenv.obs_indices as indices


class ExampleRandomAgent:
    def __init__(self):
        self.rewards = [0]
        self.total_reward = 0
        self.id = 'rando'

    def get_action(self, observation):
        # If the hand is over, the environment is asking for dummy actions to distribute final rewards.
        # This means that the action is a don't care, and will be ignored by the environment.
        # This also means, that the observation does not correspond to any meaningful choice to be taken,
        # and it should be ignored as well.
        if not observation[indices.HAND_IS_OVER]:
            valid_actions = np.argwhere(observation[indices.VALID_ACTIONS] == 1).flatten()
            valid_bet_low = observation[indices.VALID_BET_LOW]
            valid_bet_high = observation[indices.VALID_BET_HIGH]
            chosen_action = PlayerAction(np.random.choice(valid_actions))
            bet_size = 0
            if chosen_action is PlayerAction.BET:
                bet_size = np.random.uniform(valid_bet_low, valid_bet_high)
            table_action = Action(chosen_action, bet_size)
        else:
            table_action = Action(PlayerAction.CHECK, 0)
        return table_action

    def reset(self):
        self.total_reward += sum(self.rewards)
        self.rewards = [0]


class Agent:
    def __init__(self, model):
        self.observations = np.zeros([SEQUENCE_LENGTH, 60], dtype=np.float32)
        self.obs_count = np.zeros(1, dtype=np.int32)
        self.rewards = [0]
        self.total_reward = 0
        self.id = 'policy%d' % model
        self.model = model
        self.options = [('grpc.max_send_message_length', 383778839)]
        self.channel = grpc.insecure_channel('localhost:50050', options=self.options)
        self.stub = RL_pb2_grpc.ActorStub(self.channel)

    def get_action(self, observation):
        # Only save valid observations
        if obs[indices.HAND_IS_OVER]:
            if self.obs_count[0] < SEQUENCE_LENGTH:
                self.observations[self.obs_count[0]] = observation
                self.obs_count[0] += 1
            else:
                self.observations[0:SEQUENCE_LENGTH-1] = self.observations[1:SEQUENCE_LENGTH]
                self.observations[-1] = observation

            observations_bytes = np.ndarray.tobytes(self.observations)
            player_obs_count_bytes = np.ndarray.tobytes(self.obs_count)
            obs_proto = Observation(player=0, observations=observations_bytes, observation_counts=player_obs_count_bytes, shape=1, sequence_length=SEQUENCE_LENGTH)
            response = self.stub.GetStrategies(obs_proto)
            action_preds = np.frombuffer(response.action_prediction, dtype=np.float32).reshape(N_ACTIONS)
            bet_preds = np.frombuffer(response.bet_prediction, dtype=np.float32).reshape(N_BET_BUCKETS)

            pi_action = np.zeros(4)
            pi_bet = np.zeros(4)
            valid_actions = obs[indices.VALID_ACTIONS]
            min_bet = obs[indices.VALID_BET_LOW]
            max_bet = obs[indices.VALID_BET_HIGH]
            bet_sizes = np.concatenate(
                [np.array([min_bet, (min_bet + max_bet) / 2, max_bet]), BET_BUCKETS * obs[indices.POT_SIZE]])
            invalid_bets = np.logical_or(min_bet > bet_sizes, bet_sizes > max_bet)

            action_preds = np.maximum(action_preds, np.zeros_like(action_preds), dtype=np.float32)
            # Mask invalid actions and make new probabilities sum to 1
            action_preds[np.argwhere(valid_actions == 0)] = 0
            if np.count_nonzero(action_preds) == 0:
                pi_action[np.argwhere(valid_actions == 1)] = 1 / np.count_nonzero(valid_actions)
            else:
                pi_action = action_preds / action_preds.sum()

            bet_preds = np.maximum(bet_preds, np.zeros_like(bet_preds), dtype=np.float32)
            # Mask invalid bets and make new probabilities sum to 1
            bet_preds[invalid_bets] = 0
            if np.count_nonzero(bet_preds) == 0:
                pi_bet[np.logical_not(invalid_bets)] = 1 / np.count_nonzero(np.logical_not(invalid_bets))
            else:
                pi_bet = bet_preds / bet_preds.sum()

            action = np.random.choice(action_list, p=pi_action)
            action = action_list[action]
            bet_size = 0
            if action == PlayerAction.BET:
                bet_size = np.random.choice(bet_sizes, p=pi_bet)
            table_action = Action(action, bet_size)
        else:
            # Hand is over and we are only collecting final rewards, actions are ignored,
            # so send a dummy action without recording it
            table_action = Action(PlayerAction.CHECK, 0)
        return table_action

    def reset(self):
        self.observations = np.zeros([5, 60], dtype=np.float32)
        self.obs_count[0] = 0
        self.total_reward += sum(self.rewards)
        self.rewards = [0]

active_players = 2
agents = [Agent(0)]
agents.append(ExampleRandomAgent())
random_seed = 1
low_stack_bbs = 50
high_stack_bbs = 200
hh_location = '../hands/'
invalid_penalty = 0
table = Table(active_players, low_stack_bbs, high_stack_bbs, hh_location)
table.hand_history_enabled = True

iteration = 1
total_hands = 0
while True:
    if iteration == 1000:
        # table.hand_history_enabled = True
        iteration = 0
        print(total_hands)
        print("trained, bb/100 hands", agents[0].total_reward / (total_hands / 100))
        print("random, bb/100 hands", agents[1].total_reward / (total_hands / 100))

    obs = table.reset()
    acting_player = int(obs[indices.ACTING_PLAYER])
    while True:
        action = agents[acting_player].get_action(obs)
        obs, reward, done, _ = table.step(action)
        # If the reward is delayed, we are collecting end of game rewards by feeding in dummy actions
        delayed_reward = obs[indices.DELAYED_REWARD]

        if delayed_reward:
            # If the reward is delayed, the action we just took was a don't care,
            # and the reward corresponds to the last valid action taken
            agents[acting_player].rewards[-1] += reward
        else:
            # Otherwise the reward corresponds to the action we just took
            agents[acting_player].rewards.append(reward)
        if done:
            for agent in agents:
                agent.reset()
            break
        acting_player = int(obs[indices.ACTING_PLAYER])
    iteration += 1
    total_hands += 1
