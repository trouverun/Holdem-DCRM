import logging
import argparse
import grpc
import time
import numpy as np
from multiprocessing import Process, Event, Lock
from rpc import RL_pb2_grpc
from rpc.RL_pb2 import Who, Empty, Identifier
from rpc.RL_pb2_grpc import LearnerStub, MasterServicer, MasterStub, SlaveServicer, SlaveStub
from concurrent.futures import ThreadPoolExecutor
from server.server import serve
from client.client import deep_cfr
from config import CLIENT_HOSTS, SERVER_HOST, N_PLAYERS, N_ITERATIONS, K, K_PER_WORKER, TRAVERSALS_PER_PROCESS, N_PROCESSES_PER_WORKER

modes = ['master', 'slave', 'server', 'single_machine']


class Master(MasterServicer):
    def __init__(self):
        self.workers_ready = Event()
        self.worker_lock = Lock()
        self.worker_traversals = np.zeros(len(CLIENT_HOSTS))
        self.workers_completed = 0
        Process(target=self._main_loop, args=()).start()

    def _main_loop(self):
        options = [('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)]
        worker_stubs = [SlaveStub(grpc.insecure_channel(host, options)) for host in CLIENT_HOSTS]

        for identifier, worker in enumerate(worker_stubs):
            worker.SetIdentifier(Identifier(identifier))
        learner_stub = LearnerStub(grpc.insecure_channel(SERVER_HOST, options))

        for iteration in range(N_ITERATIONS):
            start = time.time()
            for player in range(N_PLAYERS):
                for worker in worker_stubs:
                    worker.RunIterations(K_PER_WORKER)
                self.workers_ready.wait(timeout=None)
                logging.info("Training regrets for player %d" % player)
                learner_stub.TrainRegrets(Who(player=player))
            logging.info("Training strategy for iteration %d" % iteration)
            learner_stub.TrainStrategy(Empty())
            end = time.time()
            logging.info("One iteration took %fs" % (end - start))

    def CompleteTraversal(self, request, context):
        self.worker_lock.acquire()
        self.worker_traversals[request.identifier] += 1
        logging.info("%d / %d traversals completed" % (self.worker_traversals.sum(), K))
        if self.worker_traversals[request.identifier] == K_PER_WORKER:
            self.workers_completed += 1
            if self.workers_completed == len(CLIENT_HOSTS):
                self.workers_ready.set()
        self.worker_lock.release()
        return Empty()


class Slave(SlaveServicer):
    def __init__(self):
        self.identifier = None

    def SetIdentifier(self, request, context):
        self.identifier = request.identifier
        return Empty()

    def RunIterations(self, request, context):
        return Empty()

    def RunEvaluations(self, request, context):
        pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str)
    parser.add_argument('-hosts', type=str, nargs='+')
    args = parser.parse_args()
    options = [('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)]

    if args.mode == 'master':
        server = grpc.server(ThreadPoolExecutor(max_workers=16), options=options)
        RL_pb2_grpc.add_MasterServicer_to_server(Master(), server)
        server.add_insecure_port(args.hosts[0])
        server.start()
        logging.info("Master server serving at %s", args.hosts[0])
        server.wait_for_termination()
    elif args.mode == 'slave':
        server = grpc.server(ThreadPoolExecutor(max_workers=16), options=options)
        RL_pb2_grpc.add_MasterServicer_to_server(Slave(), server)
        server.add_insecure_port(args.hosts[0])
        server.start()
        logging.info("Slave server serving at %s", args.hosts[0])
        server.wait_for_termination()
    elif args.mode == 'server':
        serve(args.hosts)
    else:
        Process(target=serve, args=(SERVER_HOST,)).start()
        time.sleep(5)
        learner_stub = LearnerStub(grpc.insecure_channel(SERVER_HOST[1], options))
        for iteration in range(N_ITERATIONS):
            start = time.time()
            for player in range(N_PLAYERS):
                cfr_traverse(player, K, TRAVERSALS_PER_PROCESS, N_PROCESSES_PER_WORKER)
                logging.info("Training regrets for player %d" % player)
                learner_stub.TrainRegrets(Who(player=player))
            logging.info("Training strategy for iteration %d" % iteration)
            learner_stub.TrainStrategy(Empty())
            end = time.time()
            logging.info("One iteration took %fs" % (end - start))
