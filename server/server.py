import grpc
import logging
from rpc import RL_pb2_grpc
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from multiprocessing import Event
from server.ActorServer import Actor
from server.LearnerServer import RegretLearner, StrategyLearner
from config import N_THREADPOOL_WORKERS, GLOBAL_STRATEGY_HOST, REGRET_HOST_PLAYER_MAP, ACTOR_HOST_PLAYER_MAP


def _run_actor_server(bind_address, player_list, gpu_lock):
    options = [('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)]
    server = grpc.server(ThreadPoolExecutor(max_workers=N_THREADPOOL_WORKERS), options=options)
    RL_pb2_grpc.add_ActorServicer_to_server(Actor(player_list, gpu_lock), server)
    server.add_insecure_port(bind_address)
    server.start()
    logging.info("Actor server serving at %s", bind_address)
    server.wait_for_termination()


def _run_regret_learner_server(bind_address, player_list, gpu_lock, ready):
    options = [('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)]
    server = grpc.server(ThreadPoolExecutor(max_workers=N_THREADPOOL_WORKERS), options=options)
    RL_pb2_grpc.add_RegretLearnerServicer_to_server(RegretLearner(player_list, gpu_lock, ready), server)
    server.add_insecure_port(bind_address)
    server.start()
    logging.info("Regret learner server serving at %s", bind_address)
    server.wait_for_termination()


def _run_strategy_learner_server(bind_address, gpu_lock, ready):
    options = [('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)]
    server = grpc.server(ThreadPoolExecutor(max_workers=N_THREADPOOL_WORKERS), options=options)
    RL_pb2_grpc.add_StrategyLearnerServicer_to_server(StrategyLearner(gpu_lock, ready), server)
    server.add_insecure_port(bind_address)
    server.start()
    logging.info("Strategy learner serving at %s", bind_address)
    server.wait_for_termination()


def serve(hosts):
    regret_processes = []
    reg_readies = []
    actor_processes = []
    strat_ready = None
    strategy_process = None
    gpu_lock = Lock()

    for host in hosts:
        if host in REGRET_HOST_PLAYER_MAP.keys():
            reg_readies.append(Event())
            regret_processes.append(multiprocessing.Process(target=_run_regret_learner_server,
                                                    args=(host, REGRET_HOST_PLAYER_MAP[host], gpu_lock, reg_readies[-1])))
        elif host in ACTOR_HOST_PLAYER_MAP.keys():
            actor_processes.append(multiprocessing.Process(target=_run_actor_server, args=(host, ACTOR_HOST_PLAYER_MAP[host], gpu_lock)))
        elif host == GLOBAL_STRATEGY_HOST:
            strat_ready = Event()
            strategy_process = multiprocessing.Process(target=_run_strategy_learner_server, args=(host, gpu_lock, strat_ready))
        else:
            raise ValueError("Unrecognized hostname given to serve(), check config file.")

    for i, p in enumerate(regret_processes):
        p.start()
        reg_readies[i].wait()
    if strategy_process is not None:
        strategy_process.start()
        strat_ready.wait()
    for p in actor_processes:
        p.start()