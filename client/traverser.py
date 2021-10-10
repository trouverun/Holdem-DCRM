import grpc
import gc
import numpy as np
import logging
from rpc.RL_pb2_grpc import ActorStub, RegretLearnerStub, StrategyLearnerStub, MasterStub
from rpc.RL_pb2 import Observation, SampledData, Empty
from client.batchedtraversal import BatchedTraversal
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
from config import N_PLAYERS, N_BET_BUCKETS, CLIENT_SAMPLES_MIN_BATCH_SIZE, SEQUENCE_LENGTH, N_ACTIONS, PLAYER_ACTOR_HOST_MAP, \
    PLAYER_REGRET_HOST_MAP, REGRET_HOST_PLAYER_MAP, ACTOR_HOST_PLAYER_MAP, GLOBAL_STRATEGY_HOST, MASTER_HOST


def clear_traverse_queue_process(regret_ques, strategy_ques):
    options = [('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)]
    regret_channels = {host: grpc.insecure_channel(host, options) for host in REGRET_HOST_PLAYER_MAP.keys()}
    strategy_channel = grpc.insecure_channel(GLOBAL_STRATEGY_HOST, options)
    for player in range(N_PLAYERS):
        Thread(target=clear_queue_thread, args=(player, regret_channels[PLAYER_REGRET_HOST_MAP[player]], "regret", regret_ques[player])).start()
        Thread(target=clear_queue_thread, args=(player, strategy_channel, "strategy", strategy_ques[player])).start()


def clear_queue_thread(player, channel, que_type, que):
    logging.info("Started regret/strategy queue clearing thread for player %d" % player)
    if que_type == 'regret':
        stub = RegretLearnerStub(channel)
    elif que_type == 'strategy':
        stub = StrategyLearnerStub(channel)
    else:
        raise ValueError("Invalid type for queue stub")
    observations = []
    counts = []
    actions = []
    bets = []
    while True:
        obs, count, action, bet = que.get()
        observations.extend(obs)
        counts.extend(count)
        actions.extend(action)
        bets.extend(bet)
        if len(observations) >= CLIENT_SAMPLES_MIN_BATCH_SIZE:
            observations_bytes = np.asarray(observations, dtype=np.float32).tobytes()
            player_obs_count_bytes = np.asarray(counts, dtype=np.int32).tobytes()
            action_bytes = np.asarray(actions, dtype=np.float32).tobytes()
            bet_bytes = np.asarray(bets, dtype=np.float32).tobytes()
            sampled_proto = SampledData(player=player, observations=observations_bytes, observation_counts=player_obs_count_bytes,
                                        action_data=action_bytes, bet_data=bet_bytes, shape=len(observations), sequence_length=SEQUENCE_LENGTH)
            if que_type == "regret":
                _ = stub.AddRegrets(sampled_proto)
            elif que_type == "strategy":
                _ = stub.AddStrategies(sampled_proto)
            observations = []
            counts = []
            actions = []
            bets = []
            gc.collect()


def send_player_inference_batch(args):
    channel, player, observations, observation_counts, inference_type = args
    n_items = len(observations)
    if n_items > 0:
        stub = ActorStub(channel)
        observations_bytes = np.ndarray.tobytes(np.asarray(observations, dtype=np.float32))
        player_obs_count_bytes = np.ndarray.tobytes(np.expand_dims(np.asarray(observation_counts, dtype=np.int32), 0))
        obs_proto = Observation(player=player, observations=observations_bytes, observation_counts=player_obs_count_bytes, shape=n_items,
                                sequence_length=SEQUENCE_LENGTH)

        if inference_type == 'regret':
            response = stub.GetRegrets(obs_proto)
        elif inference_type == 'strategy':
            response = stub.GetStrategies(obs_proto)
        else:
            raise ValueError("Invalid value for type of inference")

        action_preds = np.frombuffer(response.action_prediction, dtype=np.float32).reshape(n_items, N_ACTIONS)
        bet_preds = np.frombuffer(response.bet_prediction, dtype=np.float32).reshape(n_items, N_BET_BUCKETS)
        return action_preds, bet_preds
    return np.empty(0), np.empty(0)


def traverse_process(traversals_per_process, traverser, regret_que, strategy_que):
    options = [('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)]
    master_channel = grpc.insecure_channel(MASTER_HOST, options)
    master_stub = MasterStub(master_channel)
    channels = {host: grpc.insecure_channel(host, options) for host in ACTOR_HOST_PLAYER_MAP.keys()}
    bt = BatchedTraversal(traversals_per_process, traverser, regret_que, strategy_que)
    while True:
        response = master_stub.RequestTraversal(Empty())
        if response.value == -1:
            master_stub.ExitTraversalPool(Empty())
            break
        obs, obs_counts, mapping = bt.reset()
        while True:
            n_items = 0
            for player in range(N_PLAYERS):
                n_items += len(obs[player])
            if n_items == 0:
                break
            with ThreadPoolExecutor(max_workers=N_PLAYERS) as executor:
                player_channels = [channels[PLAYER_ACTOR_HOST_MAP[player]] for player in range(N_PLAYERS)]
                arg = list(zip(player_channels, list(range(N_PLAYERS)), obs, obs_counts, ['regret'] * N_PLAYERS))
                result = executor.map(send_player_inference_batch, arg)
            action_regrets = [np.empty(0) for _ in range(N_PLAYERS)]
            bet_regrets = [np.empty(0) for _ in range(N_PLAYERS)]
            for player, regrets in enumerate(result):
                ar, br = regrets
                action_regrets[player] = ar
                bet_regrets[player] = br
            obs, obs_counts, mapping = bt.step(action_regrets, bet_regrets, mapping)
