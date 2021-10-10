import logging
from multiprocessing import Process, Queue
from rpc.RL_pb2 import Empty
from rpc.RL_pb2_grpc import SlaveServicer
from config import N_PLAYERS, N_TRAVERSE_QUE_PROCESS, N_PPO_EVAL_QUE_PROCESSES, N_CONC_TRAVERSALS_PER_PROCESS, N_TRAVERSE_PROCESSES, \
    CLIENT_SAMPLES_MIN_BATCH_SIZE, PPO_EVAL_REWARD_SAMPLES_MIN_BATCH_SIZE, PPO_EVAL_TRAJECTORY_SAMPLES_MIN_BATCH_SIZE, \
    N_CONC_PPO_EVALUATIONS_PER_PROCESS, N_PPO_EVAL_PROCESSES, USE_PPO_EVALUATION, N_MCTS_PROCESSES
from client.traverser import clear_traverse_queue_process, traverse_process
from client.eval.ppo.ppo_evaluator import clear_eval_queue_process, exploitability_approximation_process
from client.eval.mcts.mcts_evaluator import run_evaluations


class Slave(SlaveServicer):
    def __init__(self):
        self.identifier = None
        self.traverser_que = Queue()
        self.evaluation_que = Queue()
        self.traverse_que_processes = []
        self.evaluate_que_processes = []
        self.traverse_processes = []
        self.evaluate_processes = []
        Process(target=self._traverse_background_process, args=()).start()
        Process(target=self._evaluate_background_process, args=()).start()

    def _traverse_background_process(self):
        while True:
            traverser = self.traverser_que.get()
            regret_ques = [Queue(maxsize=5 * CLIENT_SAMPLES_MIN_BATCH_SIZE) for _ in range(N_PLAYERS)]
            strategy_ques = [Queue(maxsize=5 * CLIENT_SAMPLES_MIN_BATCH_SIZE) for _ in range(N_PLAYERS)]
            self.traverse_que_processes = [Process(target=clear_traverse_queue_process, args=(regret_ques, strategy_ques)) for _ in
                                           range(N_TRAVERSE_QUE_PROCESS)]
            for p in self.traverse_que_processes:
                p.start()
            process_count = 0
            self.traverse_processes = [
                Process(target=traverse_process, args=(N_CONC_TRAVERSALS_PER_PROCESS, traverser, regret_ques[traverser], strategy_ques))
                for _ in range(N_TRAVERSE_PROCESSES)
            ]
            for p in self.traverse_processes:
                process_count += 1
                logging.info("starting traversal process %d" % process_count)
                p.start()
            for p in self.traverse_processes:
                p.join()
                process_count -= 1
                logging.info("joined traversal process %d" % process_count)
            for p in self.traverse_que_processes:
                p.terminate()

    def _evaluate_background_process(self):
        while True:
            iteration = self.evaluation_que.get()
            if USE_PPO_EVALUATION:
                trajectory_queue = Queue(maxsize=5 * PPO_EVAL_TRAJECTORY_SAMPLES_MIN_BATCH_SIZE)
                reward_que = Queue(maxsize=5 * PPO_EVAL_REWARD_SAMPLES_MIN_BATCH_SIZE)
                self.evaluate_que_processes = [Process(target=clear_eval_queue_process, args=(trajectory_queue, reward_que)) for _ in
                                               range(N_PPO_EVAL_QUE_PROCESSES)]
                for p in self.evaluate_que_processes:
                    p.start()
                self.evaluate_processes = [
                    Process(target=exploitability_approximation_process, args=(N_CONC_PPO_EVALUATIONS_PER_PROCESS, 0, trajectory_queue, reward_que))
                    for _ in range(N_PPO_EVAL_PROCESSES)
                ]
            else:
                self.evaluate_processes = [
                    Process(target=run_evaluations, args=(iteration,))
                    for _ in range(N_MCTS_PROCESSES)
                ]
            process_count = 0
            for p in self.evaluate_processes:
                process_count += 1
                logging.info("starting evaluation process %d" % process_count)
                p.start()
            for p in self.evaluate_processes:
                p.join()
                process_count -= 1
                logging.info("joined evaluation process %d" % process_count)
            if USE_PPO_EVALUATION:
                for p in self.evaluate_que_processes:
                    p.terminate()

    def SetIdentifier(self, request, context):
        self.identifier = request.value
        return Empty()

    def RunTraversals(self, request, context):
        self.traverser_que.put(request.value)
        return Empty()

    def RunEvaluations(self, request, context):
        self.evaluation_que.put(request.value)
        return Empty()