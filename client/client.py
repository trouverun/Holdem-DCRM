import argparse
import grpc
import faulthandler
import numpy as np
import logging
import time
from rpc.RL_pb2_grpc import ActorStub, LearnerStub
from rpc.RL_pb2 import Observation, SampledData, Empty, Who
from BatchedTraversal import BatchedTraversal
from multiprocessing import Process, Queue
from config import N_PLAYERS, N_BET_BUCKETS, CLIENT_SAMPLES_BATCH_SIZE, SEQUENCE_LENGTH, N_ACTIONS
from concurrent.futures import ThreadPoolExecutor


def send_player_batch(args):
    channel, player, observations, observation_counts = args
    n_items = len(observations)
    if n_items > 0:
        stub = ActorStub(channel)
        observations_bytes = np.ndarray.tobytes(np.asarray(observations))
        player_obs_count_bytes = np.ndarray.tobytes(np.expand_dims(np.asarray(observation_counts, dtype=np.int32), 0))
        obs_proto = Observation(player=player, observations=observations_bytes, observation_counts=player_obs_count_bytes, shape=n_items,
                                sequence_length=SEQUENCE_LENGTH)
        response = stub.GetRegrets(obs_proto)
        action_regrets = np.frombuffer(response.action_prediction, dtype=np.float32).reshape(n_items, N_ACTIONS)
        bet_regrets = np.frombuffer(response.bet_prediction, dtype=np.float32).reshape(n_items, N_BET_BUCKETS)
        return action_regrets, bet_regrets
    return np.empty(0), np.empty(0)


def traverse_process(n, channel, traversals_per_process, loops_per_process, traverser, regret_que, strategy_que):
    bt = BatchedTraversal(traversals_per_process, traverser, regret_que, strategy_que)
    for _ in range(loops_per_process):
        obs, obs_counts, mapping = bt.reset()
        while True:
            n_items = 0
            for player in range(N_PLAYERS):
                n_items += len(obs[player])
            if n_items == 0:
                break
            with ThreadPoolExecutor(max_workers=N_PLAYERS) as executor:
                arg = list(zip([channel] * N_PLAYERS, list(range(N_PLAYERS)), obs, obs_counts))
                result = executor.map(send_player_batch, arg)
            action_regrets = [np.empty(0) for _ in range(N_PLAYERS)]
            bet_regrets = [np.empty(0) for _ in range(N_PLAYERS)]
            for player, regrets in enumerate(result):
                ar, br = regrets
                action_regrets[player] = ar
                bet_regrets[player] = br
            obs, obs_counts, mapping = bt.step(action_regrets, bet_regrets, mapping)


def clear_queue_process(player, type, que):
    logging.info("Started clear queue process for player %d" % player)
    options = [('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)]
    channel = grpc.insecure_channel('localhost:50051', options)
    stub = LearnerStub(channel)
    observations = []
    counts = []
    actions = []
    bets = []
    while True:
        obs, count, action, bet = que.get()
        if obs is None:
            break
        observations.append(obs)
        counts.append(count)
        actions.append(action)
        bets.append(bet)
        if len(observations) == CLIENT_SAMPLES_BATCH_SIZE:
            observations_bytes = np.asarray(observations).tobytes()
            player_obs_count_bytes = np.asarray(counts, dtype=np.int32).tobytes()
            action_bytes = np.asarray(actions).tobytes()
            bet_bytes = np.asarray(bets).tobytes()
            sampled_proto = SampledData(player=player, observations=observations_bytes, observation_counts=player_obs_count_bytes,
                                        action_data=action_bytes, bet_data=bet_bytes, shape=CLIENT_SAMPLES_BATCH_SIZE, sequence_length=5)
            if type == "regret":
                _ = stub.AddRegrets(sampled_proto)
            elif type == "strategy":
                _ = stub.AddStrategies(sampled_proto)
            observations = []
            counts = []
            actions = []
            bets = []


def deep_cfr(iterations, k, traversals_per_process, n_processes):
    loops_per_process = int(k / (traversals_per_process*n_processes))
    options = [('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)]
    channel = grpc.insecure_channel('localhost:50051', options)
    stub_learner = LearnerStub(channel)
    inference_channels = [grpc.insecure_channel('localhost:50050', options) for _ in range(n_processes)]
    regret_ques = [Queue() for _ in range(N_PLAYERS)]
    strategy_ques = [Queue() for _ in range(N_PLAYERS)]

    # Start processes which are responsible for emptying the regret and strategy ques (sending the data to learner server)
    r_que_processors = [Process(target=clear_queue_process, args=(player, "regret", regret_ques[player])) for player in range(N_PLAYERS)]
    s_que_processors = [Process(target=clear_queue_process, args=(player, "strategy", strategy_ques[player])) for player in range(N_PLAYERS)]
    que_processors = r_que_processors + s_que_processors
    for p in que_processors:
        p.start()

    process_count = 0
    for iteration in range(iterations):
        start = time.time()
        for player in range(N_PLAYERS):
            processes = [
                Process(target=traverse_process, args=(n, inference_channels[n], traversals_per_process, loops_per_process, player,
                                                       regret_ques[player], strategy_ques[player]))
                for n in range(n_processes)
            ]
            for p in processes:
                process_count += 1
                logging.debug("starting process %d" % process_count)
                p.start()
            for p in processes:
                p.join()
                process_count -= 1
                logging.debug("joined process %d" % process_count)
            # traverse_process(0,inference_channels[0], traversals_per_process, k, player, regret_ques[player], strategy_ques[player])
            print("Training regrets for player %d" % player)
            stub_learner.TrainRegrets(Who(player=player))
        print("Training strategy for iteration %d" % iteration)
        stub_learner.TrainStrategy(Empty())
        end = time.time()
        logging.info("Run took %f" % (end - start))

    # Close the queues
    for que in regret_ques + strategy_ques:
        que.put((None, None, None, None))
    for p in que_processors:
        p.join()


if __name__ == "__main__":
    #faulthandler.enable()
    logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser()
    parser.add_argument('iterations', type=int)
    parser.add_argument('k', type=int)
    parser.add_argument('traversals_per_process', type=int)
    parser.add_argument('n_processes', type=int)
    args = parser.parse_args()
    if args.k % (args.traversals_per_process * args.n_processes) != 0:
        raise Exception("The k value needs to be divisible by the effective number of traversals per iteration "
                        "(traversals per process * number of processes)")
    deep_cfr(iterations=args.iterations, k=args.k, traversals_per_process=args.traversals_per_process, n_processes=args.n_processes)
