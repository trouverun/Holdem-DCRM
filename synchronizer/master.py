import logging
import grpc
import time
from multiprocessing import Process, Event, Lock
from threading import Thread
from rpc.RL_pb2 import IntMessage, Empty, Selection
from rpc.RL_pb2_grpc import StrategyLearnerStub, RegretLearnerStub, MasterServicer, SlaveStub, ActorStub
from config import SLAVE_HOSTS, N_PLAYERS, N_ITERATIONS, K, ACTOR_HOST_PLAYER_MAP, PLAYER_ACTOR_HOST_MAP, GLOBAL_STRATEGY_HOST, \
    REGRET_HOST_PLAYER_MAP, PLAYER_REGRET_HOST_MAP, N_TRAVERSE_PROCESSES, NUM_EVAL_LOOPS
from client.evaluator import run_evaluations


class Master(MasterServicer):
    def __init__(self):
        self.worker_traversals_ready = Event()
        self.worker_evaluations_ready = Event()
        self.traversal_counter_lock = Lock()
        self.evaluation_counter_lock = Lock()
        self.traversal_lock = Lock()
        self.worker_lock = Lock()
        self.traversals_left = K
        self.workers_remaining = len(SLAVE_HOSTS)*N_TRAVERSE_PROCESSES
        Thread(target=self._main_loop, args=()).start()

    def _main_loop(self):
        options = [('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)]
        worker_stubs = [SlaveStub(grpc.insecure_channel(host, options)) for host in SLAVE_HOSTS]

        for identifier, worker in enumerate(worker_stubs):
            worker.SetIdentifier(IntMessage(value=identifier))

        actor_channels = {host: grpc.insecure_channel(host, options) for host in ACTOR_HOST_PLAYER_MAP.keys()}
        regret_channels = {host: grpc.insecure_channel(host, options) for host in REGRET_HOST_PLAYER_MAP.keys()}
        strategy_channel = grpc.insecure_channel(GLOBAL_STRATEGY_HOST)
        strategy_stub = StrategyLearnerStub(strategy_channel)

        for iteration in range(N_ITERATIONS):
            logging.info("Starting iteration %d" % iteration)
            start = time.time()
            for player in range(N_PLAYERS):
                regret_stub = RegretLearnerStub(regret_channels[PLAYER_REGRET_HOST_MAP[player]])
                self.traversal_lock.acquire()
                self.traversals_left = K
                self.traversal_lock.release()
                self.worker_lock.acquire()
                self.worker_traversals_ready.clear()
                self.workers_remaining = len(SLAVE_HOSTS)*N_TRAVERSE_PROCESSES
                self.worker_lock.release()
                for worker in worker_stubs:
                    worker.RunTraversals(IntMessage(value=player))
                self.worker_traversals_ready.wait(timeout=None)
                logging.info("Training regrets for player %d" % player)
                regret_stub.TrainRegrets(IntMessage(value=player))
            logging.info("Training strategy for iteration %d" % iteration)
            strategy_stub.TrainStrategy(Empty())

            response = strategy_stub.AvailableStrategies(Empty())
            n_strategies = response.value
            for player in range(1, N_PLAYERS):
                actor_stub = ActorStub(actor_channels[PLAYER_ACTOR_HOST_MAP[0]])
                actor_stub.SetStrategy(Selection(player=0, strategy_version=n_strategies))
            avg_exploitability = run_evaluations(NUM_EVAL_LOOPS)
            logging.info("Average exploitability: %f" % avg_exploitability)
            end = time.time()
            logging.info("One iteration took %fs" % (end - start))

    def RequestTraversal(self, request, context):
        self.traversal_lock.acquire()
        retval = -1
        if self.traversals_left > 0:
            self.traversals_left -= 1
            logging.info("%d traversal jobs remaining" % self.traversals_left)
            retval = self.traversals_left
        self.traversal_lock.release()
        return IntMessage(value=retval)

    def ExitWorkerPool(self, request, context):
        self.worker_lock.acquire()
        self.workers_remaining -= 1
        if self.workers_remaining == 0:
            self.worker_traversals_ready.set()
        self.worker_lock.release()
        return Empty()