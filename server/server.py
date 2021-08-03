import grpc
import logging
from rpc import RL_pb2_grpc
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from multiprocessing import Event
from ActorServer import Actor
from LearnerServer import RegretLearner, StrategyLearner


def _run_actor_server(bind_address, player_list, gpu_lock):
    options = [('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)]
    server = grpc.server(ThreadPoolExecutor(max_workers=8), options=options)
    RL_pb2_grpc.add_ActorServicer_to_server(Actor(player_list, gpu_lock), server)
    server.add_insecure_port(bind_address)
    server.start()
    logging.info("Actor server serving at %s", bind_address)
    server.wait_for_termination()


def _run_regret_learner_server(bind_address, player_list, gpu_lock, ready):
    options = [('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)]
    server = grpc.server(ThreadPoolExecutor(max_workers=8), options=options)
    RL_pb2_grpc.add_RegretLearnerServicer_to_server(RegretLearner(player_list, gpu_lock, ready), server)
    server.add_insecure_port(bind_address)
    server.start()
    logging.info("Regret learner server serving at %s", bind_address)
    server.wait_for_termination()


def _run_strategy_learner_server(bind_address, gpu_lock, ready):
    options = [('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)]
    server = grpc.server(ThreadPoolExecutor(max_workers=8), options=options)
    RL_pb2_grpc.add_StrategyLearnerServicer_to_server(StrategyLearner(gpu_lock, ready), server)
    server.add_insecure_port(bind_address)
    server.start()
    logging.info("Strategy learner serving at %s", bind_address)
    server.wait_for_termination()


def serve():
    processes = []
    gpu_lock = Lock()
    readies = [Event(), Event()]
    processes.append(multiprocessing.Process(target=_run_regret_learner_server, args=("localhost:50051", [0, 1], gpu_lock, readies[0])))
    processes.append(multiprocessing.Process(target=_run_strategy_learner_server, args=("localhost:50052", gpu_lock, readies[1])))
    processes.append(multiprocessing.Process(target=_run_actor_server, args=("localhost:50053", [0, 1], gpu_lock,)))
    for i, p in enumerate(processes):
        p.start()
        if 0 <= i < 2:
            readies[i].wait()
    for p in processes:
        p.join()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    serve()