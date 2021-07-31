import grpc
import argparse
import numpy as np
import logging
import time
from rpc.RL_pb2_grpc import ActorStub, LearnerStub
from rpc.RL_pb2 import Observation, SampledData, Empty, IntMessage, Selection
from batchedtraversal import BatchedTraversal
from multiprocessing import Process, Queue
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
from pokerenv.table import Table
from batchedenv import BatchedEnvironment
from config import N_PLAYERS, N_BET_BUCKETS, CLIENT_SAMPLES_MIN_BATCH_SIZE, SEQUENCE_LENGTH, N_ACTIONS, N_QUE_PROCESS, EVAL_ENVS_PER_PROCESS, \
    N_EVAL_HANDS, N_EVAL_PROCESSES, N_TRAVERSE_PROCESSES, N_CONC_TRAVERSALS_PER_PROCESS, HH_LOCATION


def clear_queue_thread(player, channel, type, que):
    logging.info("Started regret/strategy queue clearing thread for player %d" % player)
    stub = LearnerStub(channel)
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
            observations_bytes = np.asarray(observations).tobytes()
            player_obs_count_bytes = np.asarray(counts, dtype=np.int32).tobytes()
            action_bytes = np.asarray(actions).tobytes()
            bet_bytes = np.asarray(bets).tobytes()
            sampled_proto = SampledData(player=player, observations=observations_bytes, observation_counts=player_obs_count_bytes,
                                        action_data=action_bytes, bet_data=bet_bytes, shape=len(observations), sequence_length=5)
            if type == "regret":
                _ = stub.AddRegrets(sampled_proto)
            elif type == "strategy":
                _ = stub.AddStrategies(sampled_proto)
            observations = []
            counts = []
            actions = []
            bets = []


def clear_queue_process(regret_ques, strategy_ques):
    options = [('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)]
    channel = grpc.insecure_channel('localhost:50051', options)
    for player in range(N_PLAYERS):
        Thread(target=clear_queue_thread, args=(player, channel, "regret", regret_ques[player])).start()
        Thread(target=clear_queue_thread, args=(player, channel, "strategy", strategy_ques[player])).start()


# Send a batch of observations to an inference server and returns the regrets
def send_player_inference_batch(args):
    channel, player, observations, observation_counts, type = args
    n_items = len(observations)
    if n_items > 0:
        stub = ActorStub(channel)
        observations_bytes = np.ndarray.tobytes(np.asarray(observations))
        player_obs_count_bytes = np.ndarray.tobytes(np.expand_dims(np.asarray(observation_counts, dtype=np.int32), 0))
        obs_proto = Observation(player=player, observations=observations_bytes, observation_counts=player_obs_count_bytes, shape=n_items,
                                sequence_length=SEQUENCE_LENGTH)
        if type == 'regret':
            response = stub.GetRegrets(obs_proto)
        elif type == 'strategy':
            response = stub.GetStrategies(obs_proto)
        else:
            raise ValueError("Invalid value for type of inference")

        action_preds = np.frombuffer(response.action_prediction, dtype=np.float32).reshape(n_items, N_ACTIONS)
        bet_preds = np.frombuffer(response.bet_prediction, dtype=np.float32).reshape(n_items, N_BET_BUCKETS)
        return action_preds, bet_preds
    return np.empty(0), np.empty(0)


# Does a specified amount of game traversals, sampling regrets and global strategies in the process
def traverse_process(channel, traversals_per_process, loops_per_process, traverser, regret_que, strategy_que, iter_que):
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
                arg = list(zip([channel] * N_PLAYERS, list(range(N_PLAYERS)), obs, obs_counts, ['regret'] * N_PLAYERS))
                result = executor.map(send_player_inference_batch, arg)
            action_regrets = [np.empty(0) for _ in range(N_PLAYERS)]
            bet_regrets = [np.empty(0) for _ in range(N_PLAYERS)]
            for player, regrets in enumerate(result):
                ar, br = regrets
                action_regrets[player] = ar
                bet_regrets[player] = br
            obs, obs_counts, mapping = bt.step(action_regrets, bet_regrets, mapping)
        iter_que.put(('complete', None))


def create_env_fn():
    env = Table(2, hand_history_location=HH_LOCATION)
    return env


# Evaluates the most recent policy against past policies
def eval_process(channel, envs_per_process, hands_per_process, que):
    hands_played = 0
    total_winnings = 0
    be = BatchedEnvironment(create_env_fn, envs_per_process)
    obs, obs_counts, mapping = be.reset()
    while hands_played < hands_per_process:
        with ThreadPoolExecutor(max_workers=N_PLAYERS) as executor:
            arg = list(zip([channel] * N_PLAYERS, list(range(N_PLAYERS)), obs, obs_counts, ['strategy'] * N_PLAYERS))
            result = executor.map(send_player_inference_batch, arg)
        p_actions = [np.empty(0) for _ in range(N_PLAYERS)]
        p_bets = [np.empty(0) for _ in range(N_PLAYERS)]
        for player, regrets in enumerate(result):
            pa, pb = regrets
            p_actions[player] = pa
            p_bets[player] = pb
        obs, obs_counts, mapping, rewards = be.step(p_actions, p_bets, mapping)
        hands_played += len(rewards)
        total_winnings += rewards.sum()
    que.put(('complete', (hands_played, total_winnings)))


def tracker_process(k, queue, result_que):
    traversals = 0
    eval_reward = 0
    current_n_eval_hands = 0
    mode = None
    while True:
        command, value = queue.get()
        if command == 'mode':
            if value not in ['traversal', 'evaluation']:
                raise ValueError("Invalid value for mode command")
            mode = value
        elif command == 'reset':
            if mode == 'traversal':
                traversals = 0
            elif mode == 'evaluation':
                result_que.put(eval_reward/(current_n_eval_hands/100))
                eval_reward = 0
                current_n_eval_hands = 0
        elif command == 'complete':
            if mode == 'traversal':
                traversals += 1
                logging.info("Traversals completed %d / %d" % (traversals, k))
            elif mode == 'evaluation':
                n_hands, reward = value
                current_n_eval_hands += n_hands
                eval_reward += reward
        elif command == 'quit':
            break
        else:
            raise ValueError("Invalid command %s received by tracker process when operating in %s mode" % (command, mode))


def traverse_loop(iter_que, traverse_channels, loops_per_process, regret_ques, strategy_ques, stub_learner, iteration):
    process_count = 0
    for player in range(N_PLAYERS):
        iter_que.put(('mode', 'traversal'))
        traverse_processes = [
            Process(target=traverse_process, args=(traverse_channels[n], N_CONC_TRAVERSALS_PER_PROCESS, loops_per_process, player,
                                                   regret_ques[player], strategy_ques, iter_que))
            for n in range(N_TRAVERSE_PROCESSES)
        ]
        for p in traverse_processes:
            process_count += 1
            logging.debug("starting traversal process %d" % process_count)
            p.start()
        for p in traverse_processes:
            p.join()
            process_count -= 1
            logging.debug("joined traversal process %d" % process_count)
        logging.info("Training regrets for player %d" % player)
        stub_learner.TrainRegrets(IntMessage(value=player))
        iter_que.put(('reset', None))
    logging.info("Training strategy for iteration %d" % iteration)
    stub_learner.TrainStrategy(Empty())


def eval_loop(stub_learner, iter_que, eval_channels, process_count, options, result_que):
    evaluation_names = ['previous', 'last3', 'random5']
    a_channel = grpc.insecure_channel('localhost:50050', options)
    stub_actor = ActorStub(a_channel)

    response = stub_learner.AvailableStrategies(Empty())
    n_strategies = response.value + 1
    stub_actor.SetStrategy(Selection(player=0, strategy_version=n_strategies - 1))
    iter_que.put(('mode', 'evaluation'))
    for evaluation_name in evaluation_names:
        logging.info("Evaluating new strategy against: '%s'" % evaluation_name)
        loops = 1
        if evaluation_name == 'last3':
            if n_strategies < 4:
                continue
            loops = 3
            strategies = np.arange(n_strategies - 4, n_strategies)
        elif evaluation_name == 'random5':
            if n_strategies < 6:
                continue
            loops = 5
            strategies = np.random.choice(np.arange(n_strategies), 5, replace=False)
        elif evaluation_name == 'previous':
            if n_strategies < 2:
                continue
            strategies = [n_strategies - 2]
        else:
            raise Exception("Eval name not recognized")
        rewards = np.zeros(loops)
        for loop in range(loops):
            for player in range(N_PLAYERS):
                if player == 0:
                    continue
                stub_actor.SetStrategy(Selection(player=player, strategy_version=strategies[loop]))
            # Destroy parent grpc to make multiprocessing work correctly
            a_channel.close()
            eval_processes = [
                Process(target=eval_process,
                        args=(eval_channels[n], EVAL_ENVS_PER_PROCESS, int(N_EVAL_HANDS / N_EVAL_PROCESSES), iter_que))
                for n in range(N_EVAL_PROCESSES)
            ]
            for p in eval_processes:
                process_count += 1
                logging.debug("starting evaluation process %d" % process_count)
                p.start()
            for p in eval_processes:
                p.join()
                process_count -= 1
                logging.debug("joined evaluation process %d" % process_count)
            a_channel = grpc.insecure_channel('localhost:50050', options)
            stub_actor = ActorStub(a_channel)
            iter_que.put(('reset', None))
            reward = result_que.get()
            rewards[loop] = reward
        logging.info("Evaluation %s result was average winnings of %f bb/100hands (sample size = %d)" % (
        evaluation_name, rewards.mean(), loops * N_EVAL_HANDS))


def deep_cfr(iterations, k):
    loops_per_process = int(k / (N_CONC_TRAVERSALS_PER_PROCESS*N_TRAVERSE_PROCESSES))
    options = [('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)]
    traverse_channels = [grpc.insecure_channel('localhost:50050', options) for _ in range(N_TRAVERSE_PROCESSES)]
    eval_channels = [grpc.insecure_channel('localhost:50050', options) for _ in range(N_EVAL_PROCESSES)]
    regret_ques = [Queue() for _ in range(N_PLAYERS)]
    strategy_ques = [Queue() for _ in range(N_PLAYERS)]
    iter_que = Queue()
    result_que = Queue()
    iter_track_process = Process(target=tracker_process, args=(k, iter_que, result_que))
    iter_track_process.start()
    process_count = 0

    que_processors = [Process(target=clear_queue_process, args=(regret_ques, strategy_ques)) for _ in range(N_QUE_PROCESS)]
    for p in que_processors:
        p.start()

    for iteration in range(iterations):
        start = time.time()
        l_channel = grpc.insecure_channel('localhost:50051', options)
        stub_learner = LearnerStub(l_channel)
        traverse_loop(iter_que, traverse_channels, loops_per_process, regret_ques, strategy_ques, stub_learner, iteration)
        # Destroy parent grpc to make multiprocessing work correctly
        l_channel.close()
        eval_loop(stub_learner, iter_que, eval_channels, process_count, options, result_que)
        end = time.time()
        logging.info("One iteration took %fs" % (end - start))

    iter_que.put(('quit', None))
    iter_track_process.join()
    for p in que_processors:
        p.kill()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser()
    parser.add_argument('iterations', type=int)
    parser.add_argument('k', type=int)
    args = parser.parse_args()
    if args.k % (N_CONC_TRAVERSALS_PER_PROCESS * N_TRAVERSE_PROCESSES) != 0 or N_EVAL_HANDS % (EVAL_ENVS_PER_PROCESS * N_EVAL_PROCESSES) != 0:
        raise Exception("The k value needs to be divisible by the effective number of traversals per iteration "
                        "(traversals per process * number of processes)")
    deep_cfr(args.iterations, args.k)