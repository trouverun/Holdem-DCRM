import atexit

import grpc
import os
import logging
from rpc import RL_pb2_grpc
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from multiprocessing import Event
from server.ActorServer import Actor
from server.RegretServer import RegretLearner
from server.StrategyServer import StrategyLearner
from server.EvaluationServer import EvaluationServer
from config import N_THREADPOOL_WORKERS, GLOBAL_STRATEGY_HOST, REGRET_HOST_PLAYER_MAP, ACTOR_HOST_PLAYER_MAP, GLOBAL_EVAL_HOST


def _run_actor_server(bind_address, player_list, gpu_lock):
    options = [('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)]
    server = grpc.server(ThreadPoolExecutor(max_workers=N_THREADPOOL_WORKERS), options=options)
    RL_pb2_grpc.add_ActorServicer_to_server(Actor(player_list, gpu_lock), server)
    server.add_insecure_port(bind_address)
    server.start()
    logging.info("Actor server serving at %s", bind_address)
    server.wait_for_termination()


def _run_eval_server(bind_address, gpu_lock, ready):
    options = [('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)]
    server = grpc.server(ThreadPoolExecutor(max_workers=N_THREADPOOL_WORKERS), options=options)
    RL_pb2_grpc.add_EvaluatorServicer_to_server(EvaluationServer(gpu_lock, ready), server)
    server.add_insecure_port(bind_address)
    server.start()
    logging.info("Eval server serving at %s", bind_address)
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


class Server:
    def __init__(self):
        self.regret_processes = []
        self.actor_processes = []
        self.strategy_process = None
        self.eval_process = None

    def cleanup(self):
        for p in self.regret_processes:
            p.terminate()
        for p in self.actor_processes:
            p.terminate()
        if self.strategy_process is not None:
            self.strategy_process.terminate()
        if self.eval_process is not None:
            self.eval_process.terminate()
        logging.info("server cleanup done")

    def serve(self, hosts):
        dirs = os.listdir()
        if 'states' not in dirs:
            os.makedirs('states')
        subdirs = os.listdir('states')
        if 'regret' not in subdirs:
            os.makedirs('states/regret')
        if 'strategy' not in subdirs:
            os.makedirs('states/strategy')
        if 'reservoirs' not in dirs:
            os.makedirs('reservoirs')

        reg_readies = []
        strat_ready = None
        eval_ready = None
        gpu_lock = Lock()

        for host in hosts:
            if host in REGRET_HOST_PLAYER_MAP.keys():
                reg_readies.append(Event())
                self.regret_processes.append(multiprocessing.Process(target=_run_regret_learner_server,
                                                                     args=(host, REGRET_HOST_PLAYER_MAP[host], gpu_lock, reg_readies[-1])))
            elif host in ACTOR_HOST_PLAYER_MAP.keys():
                self.actor_processes.append(multiprocessing.Process(target=_run_actor_server, args=(host, ACTOR_HOST_PLAYER_MAP[host], gpu_lock)))
            elif host == GLOBAL_STRATEGY_HOST:
                strat_ready = Event()
                self.strategy_process = multiprocessing.Process(target=_run_strategy_learner_server, args=(host, gpu_lock, strat_ready))
            elif host == GLOBAL_EVAL_HOST:
                eval_ready = Event()
                self.eval_process = multiprocessing.Process(target=_run_eval_server, args=(host, gpu_lock, eval_ready))
            else:
                raise ValueError("Unrecognized hostname given to serve(), check the config file.")

        for i, p in enumerate(self.regret_processes):
            p.start()
            reg_readies[i].wait()
        if self.strategy_process is not None:
            self.strategy_process.start()
            strat_ready.wait()
        if self.eval_process is not None:
            self.eval_process.start()
            eval_ready.wait()
        for p in self.actor_processes:
            p.start()
