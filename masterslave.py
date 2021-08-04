import logging
import grpc
import time
from multiprocessing import Process, Event, Lock, Queue
from rpc.RL_pb2 import IntMessage, Empty, Selection
from rpc.RL_pb2_grpc import StrategyLearnerStub, RegretLearnerStub, MasterServicer, SlaveServicer, SlaveStub, ActorStub
from config import SLAVE_HOSTS, N_PLAYERS, N_ITERATIONS, K, ACTOR_HOST_PLAYER_MAP, PLAYER_ACTOR_HOST_MAP, GLOBAL_STRATEGY_HOST, \
    N_QUE_PROCESS, N_CONC_TRAVERSALS_PER_PROCESS, N_TRAVERSE_PROCESSES, N_EVALUATIONS, EVAL_ENVS_PER_PROCESS, REGRET_HOST_PLAYER_MAP, PLAYER_REGRET_HOST_MAP
from client.client import clear_queue_process, traverse_process, eval_process
from threading import Thread


class Master(MasterServicer):
    def __init__(self):
        self.worker_traversals_ready = Event()
        self.worker_evaluations_ready = Event()
        self.traversal_counter_lock = Lock()
        self.evaluation_counter_lock = Lock()
        self.training_lock = Lock()
        self.traversal_completed = False
        self.evaluation_lock = Lock()
        self.eval_completed = False
        self.traversals_completed = 0
        self.evaluation_reward = 0
        self.evaluations_completed = 0
        Process(target=self._main_loop, args=()).start()

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
                self.training_lock.acquire()
                self.traversal_completed = False
                self.training_lock.release()
                for worker in worker_stubs:
                    worker.RunTraversals(IntMessage(value=player))
                self.worker_traversals_ready.wait(timeout=None)
                logging.info("Training regrets for player %d" % player)
                regret_stub.TrainRegrets(IntMessage(value=player))
            logging.info("Training strategy for iteration %d" % iteration)
            strategy_stub.TrainStrategy(Empty())

            response = strategy_stub.AvailableStrategies(Empty())
            n_strategies = response.value + 1

            for player in range(N_PLAYERS):
                actor_stub = ActorStub(actor_channels[PLAYER_ACTOR_HOST_MAP[player]])
                if player == 0:
                    actor_stub.SetStrategy(Selection(player=player, strategy_version=n_strategies-1))
                else:
                    actor_stub.SetStrategy(Selection(player=player, strategy_version=n_strategies-2))


            self.training_lock.acquire()
            self.traversal_completed = False
            self.training_lock.release()
            for worker in worker_stubs:
                worker.RunEvaluations(Empty())
            self.worker_evaluations_ready.wait(timeout=None)

            end = time.time()
            logging.info("One iteration took %fs" % (end - start))

    def CompleteTraversal(self, request, context):
        self.traversal_counter_lock.acquire()
        self.training_lock.acquire()
        traversals_left = -1
        if not self.traversal_completed:
            self.traversals_completed += 1
            logging.info("Traversals completed: %d/%d" % (self.traversals_completed, K))
            traversals_left = K - self.traversals_completed
            if not traversals_left > 0:
                self.traversal_completed = True
                self.traversals_completed = 0
                self.worker_traversals_ready.set()
        self.training_lock.release()
        self.traversal_counter_lock.release()
        return IntMessage(value=traversals_left)

    def CompleteEvaluation(self, request, context):
        self.evaluation_counter_lock.acquire()
        self.evaluation_lock.acquire()
        evaluations_left = -1
        if not self.eval_completed:
            self.evaluation_reward += request.value1
            self.evaluations_completed += request.value2
            if self.evaluations_completed % 100 == 0:
                logging.info("Evaluations completed: %d/%d, current winnings bb/100: %f" % (self.evaluations_completed, N_EVALUATIONS,
                                                                                            self.evaluation_reward/(100*self.evaluations_completed)))
            evaluations_left = N_EVALUATIONS - self.evaluations_completed
            if not evaluations_left > 0:
                logging.info("Final evaluation winnings bb/100: %f (sample size: %d)" % (self.evaluation_reward/(100*self.evaluations_completed),
                                                                                         self.evaluations_completed))
                self.eval_completed = True
                self.evaluation_reward = 0
                self.evaluations_completed = 0
                self.worker_evaluations_ready.set()
        self.evaluation_lock.release()
        self.evaluation_counter_lock.release()
        return IntMessage(value=evaluations_left)


class Slave(SlaveServicer):
    def __init__(self):
        self.identifier = None
        self.traverser_que = Queue()
        self.evaluation_que = Queue()
        Process(target=self._background_process, args=()).start()

    def _background_process(self):
        Thread(target=self._evaluate_background_thread, args=()).start()
        Thread(target=self._traverse_background_thread, args=()).start()

    def _evaluate_background_thread(self):
        while True:
            _ = self.evaluation_que.get()
            process_count = 0
            traverse_processes = [
                Process(target=eval_process, args=(EVAL_ENVS_PER_PROCESS,))
                for n in range(N_TRAVERSE_PROCESSES)
            ]
            for p in traverse_processes:
                process_count += 1
                logging.info("starting traversal process %d" % process_count)
                p.start()
            for p in traverse_processes:
                p.join()
                process_count -= 1
                logging.info("joined traversal process %d" % process_count)

    def _traverse_background_thread(self):
        regret_ques = [Queue(maxsize=1000) for _ in range(N_PLAYERS)]
        strategy_ques = [Queue(maxsize=1000) for _ in range(N_PLAYERS)]
        que_processors = [Process(target=clear_queue_process, args=(regret_ques, strategy_ques)) for _ in range(N_QUE_PROCESS)]
        for p in que_processors:
            p.start()
        while True:
            traverser = self.traverser_que.get()
            process_count = 0
            traverse_processes = [
                Process(target=traverse_process, args=(N_CONC_TRAVERSALS_PER_PROCESS, traverser, regret_ques[traverser], strategy_ques))
                for n in range(N_TRAVERSE_PROCESSES)
            ]
            for p in traverse_processes:
                process_count += 1
                logging.info("starting traversal process %d" % process_count)
                p.start()
            for p in traverse_processes:
                p.join()
                process_count -= 1
                logging.info("joined traversal process %d" % process_count)

    def SetIdentifier(self, request, context):
        self.identifier = request.value
        return Empty()

    def RunTraversals(self, request, context):
        self.traverser_que.put(request.value)
        return Empty()

    def RunEvaluations(self, request, context):
        self.evaluation_que.put(None)
        return Empty()