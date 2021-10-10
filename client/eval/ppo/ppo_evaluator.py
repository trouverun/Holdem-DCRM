import grpc
import gc
import numpy as np
import logging
import pickle
import time
from threading import Thread
from rpc.RL_pb2_grpc import ActorStub, MasterStub, EvalPPOStub
from rpc.RL_pb2 import Observation, SampledTrajectory, Empty, IntMessage, SampledRewards
from client.eval.ppo.batchedenvironment import BatchedEnvironment
from concurrent.futures import ThreadPoolExecutor
from config import N_PLAYERS, N_BET_BUCKETS, PPO_EVAL_TRAJECTORY_SAMPLES_MIN_BATCH_SIZE, PPO_EVAL_REWARD_SAMPLES_MIN_BATCH_SIZE, SEQUENCE_LENGTH, \
    N_ACTIONS, PLAYER_ACTOR_HOST_MAP, MASTER_HOST, GLOBAL_EVAL_HOST, N_EVAL_ITERATIONS, OBS_SHAPE, EVAL_PERMISSION_WAIT_TIME


def clear_eval_queue_process(trajectory_que, reward_que):
    options = [('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)]
    eval_channel = grpc.insecure_channel(GLOBAL_EVAL_HOST, options)
    master_channel = grpc.insecure_channel(MASTER_HOST, options)
    for player in range(N_PLAYERS):
        Thread(target=clear_trajectory_queue_thread, args=(eval_channel, trajectory_que)).start()
        Thread(target=clear_reward_queue_thread, args=(master_channel, reward_que)).start()


def clear_reward_queue_thread(master_channel, que):
    logging.info("Started eval reward queue clearing thread")
    stub = MasterStub(master_channel)
    rewards = []
    while True:
        reward = que.get()
        rewards.append(reward)
        if len(rewards) >= PPO_EVAL_REWARD_SAMPLES_MIN_BATCH_SIZE:
            n_samples = len(rewards)
            reward_bytes = np.asarray(rewards, dtype=np.float32).tobytes()
            _ = stub.AddPPOEvalRewards(SampledRewards(n_samples=n_samples, rewards=reward_bytes))
            rewards = []
            gc.collect()


def clear_trajectory_queue_thread(eval_channel, que):
    logging.info("Started eval trajectory queue clearing thread")
    stub = EvalPPOStub(eval_channel)
    observations = []
    counts = []
    actions = []
    bets = []
    rewards = []
    while True:
        obs, count, action, bet, reward = que.get()
        observations.extend(obs)
        counts.extend(count)
        actions.extend(action)
        bets.extend(bet)
        rewards.extend(reward)
        if len(observations) >= PPO_EVAL_TRAJECTORY_SAMPLES_MIN_BATCH_SIZE:
            observations_bytes = np.asarray(observations, dtype=np.float32).tobytes()
            player_obs_count_bytes = np.asarray(counts, dtype=np.int32).tobytes()
            action_bytes = np.asarray(actions, dtype=np.float32).tobytes()
            bet_bytes = np.asarray(bets, dtype=np.float32).tobytes()
            reward_bytes = np.asarray(rewards, dtype=np.float32).tobytes()
            sampled_proto = SampledTrajectory(player=0, observations=observations_bytes, observation_counts=player_obs_count_bytes,
                                              action_log_probs=action_bytes, bet_log_probs=bet_bytes, rewards=reward_bytes,
                                              shape=len(observations), sequence_length=SEQUENCE_LENGTH)
            _ = stub.AddExperience(sampled_proto)
            observations = []
            counts = []
            actions = []
            bets = []
            rewards = []
            gc.collect()


def send_player_inference_batch(args):
    player, channel, inference_type, observations, observation_counts = args
    n_items = len(observations)
    if n_items > 0:
        observations_bytes = np.ndarray.tobytes(np.asarray(observations, dtype=np.float32))
        player_obs_count_bytes = np.ndarray.tobytes(np.expand_dims(np.asarray(observation_counts, dtype=np.int32), 0))
        obs_proto = Observation(player=player, observations=observations_bytes, observation_counts=player_obs_count_bytes, shape=n_items,
                                sequence_length=SEQUENCE_LENGTH)
        if inference_type == 'best_response':
            ppo_stub = EvalPPOStub(channel)
            response = ppo_stub.GetStrategies(obs_proto)
        else:
            actor_stub = ActorStub(channel)
            response = actor_stub.GetStrategies(obs_proto)
        action_preds = np.frombuffer(response.action_prediction, dtype=np.float32).reshape(n_items, N_ACTIONS)
        bet_preds = np.frombuffer(response.bet_prediction, dtype=np.float32).reshape(n_items, N_BET_BUCKETS)
        return action_preds, bet_preds
    return np.empty(0), np.empty(0)


def exploitability_approximation_process(envs_per_process, evaluated_player, trajectory_que, reward_que):
    options = [('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)]
    master_channel = grpc.insecure_channel(MASTER_HOST, options)
    master_stub = MasterStub(master_channel)
    eval_channel = grpc.insecure_channel(GLOBAL_EVAL_HOST, options)
    eval_stub = EvalPPOStub(eval_channel)
    channel_map = {player: PLAYER_ACTOR_HOST_MAP[player] for player in range(N_PLAYERS) if player != evaluated_player}
    channel_map[evaluated_player] = eval_channel
    type_map = {player: 'actor' for player in range(N_PLAYERS) if player != evaluated_player}
    type_map[evaluated_player] = 'best_response'
    be = BatchedEnvironment(envs_per_process, evaluated_player, trajectory_que, reward_que)
    for i in range(N_EVAL_ITERATIONS):
        # Wait for all workers to move on to this iteration
        while True:
            response = master_stub.GetPPOCurrentIteration(Empty())
            if response.value == i:
                break
            time.sleep(EVAL_PERMISSION_WAIT_TIME)

        response = master_stub.GetPPOTableInstance(Empty())
        env = pickle.loads(response.table)
        obs = np.frombuffer(response.initial_obs, dtype=np.float32).reshape(OBS_SHAPE)
        be.set_env(env, obs)
        be.set_training_mode(True)
        obs, obs_counts, mapping, completed_env_indices = be.reset()
        response = master_stub.RequestPPOTrainingHands(IntMessage(value=len(completed_env_indices)))
        granted_hands = response.value
        envs_left = be.disable_envs(completed_env_indices[granted_hands:len(completed_env_indices)])
        while envs_left > 0:
            with ThreadPoolExecutor(max_workers=N_PLAYERS) as executor:
                channels = [channel_map[player] for player in range(N_PLAYERS)]
                types = [type_map[player] for player in range(N_PLAYERS)]
                arg = list(zip(list(range(N_PLAYERS)), channels, types, obs, obs_counts))
                result = executor.map(send_player_inference_batch, arg)
            action_preds = [np.empty(0) for _ in range(N_PLAYERS)]
            bet_preds = [np.empty(0) for _ in range(N_PLAYERS)]
            for player, regrets in enumerate(result):
                ap, bp = regrets
                action_preds[player] = ap
                bet_preds[player] = bp
            obs, obs_counts, mapping, completed_env_indices = be.step(action_preds, bet_preds, mapping)
            if len(completed_env_indices) > 0:
                response = master_stub.RequestPPOTrainingHands(IntMessage(value=len(completed_env_indices)))
                granted_hands = response.value
                envs_left = be.disable_envs(completed_env_indices[granted_hands:len(completed_env_indices)])

        _ = master_stub.ExitPPOTrainingPool(Empty())
        # Wait for all workers to move on from the training phase
        while True:
            response = master_stub.GetPPOTrainingWorkersLeft(Empty())
            if response.value == 0:
                break
            time.sleep(EVAL_PERMISSION_WAIT_TIME)
        # Also wait for all the training trajectories to be consumed in the PPO server
        while True:
            response = eval_stub.TrajectoriesLeft(Empty())
            if response.value == 0:
                break
            time.sleep(EVAL_PERMISSION_WAIT_TIME)

        be.set_training_mode(False)
        obs, obs_counts, mapping, completed_env_indices = be.reset()
        response = master_stub.RequestPPOEvalHands(IntMessage(value=len(completed_env_indices)))
        granted_hands = response.value
        envs_left = be.disable_envs(completed_env_indices[granted_hands:len(completed_env_indices)])
        while envs_left > 0:
            with ThreadPoolExecutor(max_workers=N_PLAYERS) as executor:
                channels = [channel_map[player] for player in range(N_PLAYERS)]
                types = [type_map[player] for player in range(N_PLAYERS)]
                arg = list(zip(list(range(N_PLAYERS)), channels, types, obs, obs_counts))
                result = executor.map(send_player_inference_batch, arg)
            action_preds = [np.empty(0) for _ in range(N_PLAYERS)]
            bet_preds = [np.empty(0) for _ in range(N_PLAYERS)]
            for player, regrets in enumerate(result):
                ap, bp = regrets
                action_preds[player] = ap
                bet_preds[player] = bp
            obs, obs_counts, mapping, completed_env_indices = be.step(action_preds, bet_preds, mapping)
            if len(completed_env_indices) > 0:
                response = master_stub.RequestPPOEvalHands(IntMessage(value=len(completed_env_indices)))
                granted_hands = response.value
                envs_left = be.disable_envs(completed_env_indices[granted_hands:len(completed_env_indices)])

        _ = master_stub.ExitPPOEvaluationPool(Empty())