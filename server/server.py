import time
import grpc
import logging
from rpc import RL_pb2_grpc
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from ActorServer import Actor
from LearnerServer import Learner


def _run_actor_server(bind_address, gpu_lock):
    options = [('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)]
    server = grpc.server(ThreadPoolExecutor(max_workers=32), options=options)
    RL_pb2_grpc.add_ActorServicer_to_server(Actor(gpu_lock), server)
    server.add_insecure_port(bind_address)
    server.start()
    logging.info("Server serving at %s", bind_address)
    server.wait_for_termination()


def _run_learner_server(bind_address, gpu_lock):
    options = [('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)]
    server = grpc.server(ThreadPoolExecutor(max_workers=32), options=options)
    RL_pb2_grpc.add_LearnerServicer_to_server(Learner(gpu_lock), server)
    server.add_insecure_port(bind_address)
    server.start()
    logging.info("Server serving at %s", bind_address)
    server.wait_for_termination()


def serve():
    processes = []
    gpu_lock = Lock()
    processes.append(multiprocessing.Process(target=_run_learner_server, args=("localhost:50051", gpu_lock,)))
    processes.append(multiprocessing.Process(target=_run_actor_server, args=("localhost:50050", gpu_lock,)))
    for p in processes:
        p.start()
        time.sleep(2)
    for p in processes:
        p.join()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    serve()