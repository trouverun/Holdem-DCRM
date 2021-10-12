import logging
import grpc
import time
import pickle
import os
import shutil
import numpy as np
from pokerenv.table import Table
from multiprocessing import Event, Lock
from threading import Thread
from rpc.RL_pb2 import IntMessage, Empty, Selection, TableMessage
from rpc.RL_pb2_grpc import StrategyLearnerStub, RegretLearnerStub, MasterServicer, SlaveStub, ActorStub, EvalPPOStub
from config import SLAVE_HOSTS, N_PLAYERS, N_ITERATIONS, K, ACTOR_HOST_PLAYER_MAP, PLAYER_ACTOR_HOST_MAP, GLOBAL_STRATEGY_HOST, \
    REGRET_HOST_PLAYER_MAP, PLAYER_REGRET_HOST_MAP, N_TRAVERSE_PROCESSES, N_PPO_EVAL_HANDS, N_PPO_TRAINING_HANDS, N_PPO_EVAL_PROCESSES, \
    N_EVAL_ITERATIONS, LOW_STACK_BBS, HIGH_STACK_BBS, GLOBAL_EVAL_HOST, USE_PPO_EVALUATION, N_MCTS_EVAL_HANDS, EVAL_FREQUENCY


class Master(MasterServicer):
    def __init__(self):
        self.iteration = 0
        self.traverse_workers_ready = Event()
        self.traversal_lock = Lock()
        self.traversals_left = K
        self.traverse_workers_left = len(SLAVE_HOSTS)*N_TRAVERSE_PROCESSES

        self.eval_workers_ready = Event()
        self.evaluation_lock = Lock()
        self.average_exploitability = 0
        self.total_eval_hands_played = 0

        if USE_PPO_EVALUATION:
            self.ppo_training_hands_left = N_PPO_TRAINING_HANDS
            self.ppo_evaluation_hands_left = N_PPO_EVAL_HANDS
            self.ppo_training_workers_left = len(SLAVE_HOSTS)*N_PPO_EVAL_PROCESSES
            self.ppo_eval_workers_left = len(SLAVE_HOSTS)*N_PPO_EVAL_PROCESSES
            self.ppo_eval_iterations_done = 0
            self.ppo_last_training_info = int(2*N_PPO_TRAINING_HANDS/1000)
            self.ppo_last_eval_info = int(2*N_PPO_EVAL_HANDS/1000)
            self.table = Table(N_PLAYERS, player_names={0: 'best response', 1: 'DCRM policy'}, track_single_player=True, stack_low=LOW_STACK_BBS, stack_high=HIGH_STACK_BBS)
            if 'iteration_%d' % self.iteration not in os.listdir('hands'):
                os.makedirs('hands/iteration_%d/' % self.iteration)
            dirname = 'hands/iteration_%d/eval_hand_%d/' % (self.iteration, self.ppo_eval_iterations_done+1)
            if 'eval_hand_%d' % (self.ppo_eval_iterations_done+1) in os.listdir('hands/iteration_%d' % self.iteration):
                shutil.rmtree(dirname)
            os.makedirs(dirname)
            self.table.hand_history_location = 'hands/iteration_%d/eval_hand_%d/' % (self.iteration, self.ppo_eval_iterations_done+1)
            self.table.hand_history_enabled = True
            self.initial_obs = self.table.reset()
            self.table.hand_history_enabled = False
            self.eval_channel = grpc.insecure_channel(GLOBAL_EVAL_HOST)
            self.eval_stub = EvalPPOStub(self.eval_channel)
        else:
            self.mcts_workers_left = len(SLAVE_HOSTS)
            self.mcts_hands_left = N_MCTS_EVAL_HANDS
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

        for self.iteration in range(N_ITERATIONS):
            logging.info("Starting iteration %d" % self.iteration)
            start = time.time()
            for player in range(N_PLAYERS):
                regret_stub = RegretLearnerStub(regret_channels[PLAYER_REGRET_HOST_MAP[player]])
                self.traversal_lock.acquire()
                self.traversals_left = K
                self.traverse_workers_ready.clear()
                self.traverse_workers_left = len(SLAVE_HOSTS) * N_TRAVERSE_PROCESSES
                self.traversal_lock.release()
                for worker in worker_stubs:
                    worker.RunTraversals(IntMessage(value=player))
                self.traverse_workers_ready.wait(timeout=None)
                logging.info("Training regrets for player %d" % player)
                regret_stub.TrainRegrets(IntMessage(value=player))
            logging.info("Training strategy for iteration %d" % self.iteration)
            strategy_stub.TrainStrategy(Empty())

            if self.iteration % EVAL_FREQUENCY == 0 and self.iteration != 0:
                response = strategy_stub.AvailableStrategies(Empty())
                n_strategies = response.value
                for player in range(1, N_PLAYERS):
                    actor_stub = ActorStub(actor_channels[PLAYER_ACTOR_HOST_MAP[player]])
                    actor_stub.SetStrategy(Selection(player=player, strategy_version=n_strategies))

                self.average_exploitability = 0
                self.total_eval_hands_played = 0
                if USE_PPO_EVALUATION:
                    self.training_hands_left = N_PPO_TRAINING_HANDS
                    self.evaluation_hands_left = N_PPO_EVAL_HANDS
                    self.ppo_eval_iterations_done = 0
                    self.eval_workers_ready.clear()
                    self.training_workers_left = len(SLAVE_HOSTS) * N_PPO_EVAL_PROCESSES
                    self.eval_workers_left = len(SLAVE_HOSTS) * N_PPO_EVAL_PROCESSES
                    self.ppo_last_training_info = int(2*N_PPO_TRAINING_HANDS/1000)
                    self.ppo_last_eval_info = int(2*N_PPO_EVAL_HANDS/1000)
                else:
                    self.mcts_hands_left = N_MCTS_EVAL_HANDS
                    self.mcts_workers_left = len(SLAVE_HOSTS)

                for worker in worker_stubs:
                    worker.RunEvaluations(IntMessage(value=self.iteration))
                self.eval_workers_ready.wait(timeout=None)
                logging.info(100*self.average_exploitability)
            end = time.time()
            logging.info("One iteration took %fs" % (end - start))

    def RequestTraversals(self, request, context):
        self.traversal_lock.acquire()
        granted = min(request.value, self.traversals_left)
        self.traversals_left -= granted
        if granted > 0:
            logging.info("%d traversals remaining" % self.traversals_left)
        self.traversal_lock.release()
        return IntMessage(value=granted)

    def ExitTraversalPool(self, request, context):
        self.traversal_lock.acquire()
        self.traverse_workers_left -= 1
        if self.traverse_workers_left == 0:
            self.traverse_workers_ready.set()
        self.traversal_lock.release()
        return Empty()

    def GetPPOTableInstance(self, request, context):
        # logging.info("initial obs on master", self.initial_obs)
        initial_obs_bytes = np.ndarray.tobytes(np.asarray(self.initial_obs, dtype=np.float32))
        return TableMessage(table=pickle.dumps(self.table), initial_obs=initial_obs_bytes)

    def GetPPOCurrentIteration(self, request, context):
        self.evaluation_lock.acquire()
        retval = self.ppo_eval_iterations_done
        self.evaluation_lock.release()
        return IntMessage(value=retval)

    def GetPPOTrainingWorkersLeft(self, request, context):
        self.evaluation_lock.acquire()
        retval = self.ppo_training_hands_left
        self.evaluation_lock.release()
        return IntMessage(value=retval)

    def RequestPPOTrainingHands(self, request, context):
        self.evaluation_lock.acquire()
        granted = min(request.value, self.ppo_training_hands_left)
        self.ppo_training_hands_left -= granted
        if granted > 0:
            val = int(self.ppo_training_hands_left / 1000)
            if val < self.ppo_last_training_info:
                self.ppo_last_training_info = val
                logging.info("%d training hands remaining" % self.ppo_training_hands_left)
        self.evaluation_lock.release()
        return IntMessage(value=granted)

    def RequestPPOEvalHands(self, request, context):
        self.evaluation_lock.acquire()
        granted = min(request.value, self.ppo_evaluation_hands_left)
        self.ppo_evaluation_hands_left -= granted
        if granted > 0:
            val = int(self.ppo_training_hands_left / 1000)
            if val < self.ppo_last_eval_info:
                self.ppo_last_eval_info = val
                logging.info("%d evaluation hands remaining" % self.ppo_evaluation_hands_left)
        self.evaluation_lock.release()
        return IntMessage(value=granted)

    def AddPPOEvalRewards(self, request, context):
        self.evaluation_lock.acquire()
        rewards = np.frombuffer(request.rewards, dtype=np.float32).reshape(request.n_samples)
        for reward in rewards:
            self.average_exploitability = (self.total_eval_hands_played * self.average_exploitability + reward) / (self.total_eval_hands_played + 1)
            self.total_eval_hands_played += 1
        self.evaluation_lock.release()
        return Empty()

    def ExitPPOTrainingPool(self, request, context):
        self.evaluation_lock.acquire()
        self.ppo_training_workers_left -= 1
        self.evaluation_lock.release()
        return Empty()

    def ExitPPOEvaluationPool(self, request, context):
        self.evaluation_lock.acquire()
        self.ppo_eval_workers_left -= 1
        if self.ppo_eval_workers_left == 0:
            self.ppo_eval_iterations_done += 1
            logging.info("Eval iterations: %d/%d, average exploitability: %f bb/100" %
                         (self.ppo_eval_iterations_done, N_EVAL_ITERATIONS, 100*self.average_exploitability))
            if 'iteration_%d' % self.iteration not in os.listdir('hands'):
                os.makedirs('hands/iteration_%d/' % self.iteration)
            dirname = 'hands/iteration_%d/eval_hand_%d/' % (self.iteration, self.ppo_eval_iterations_done+1)
            if 'eval_hand_%d' % (self.ppo_eval_iterations_done+1) in os.listdir('hands/iteration_%d' % self.iteration):
                shutil.rmtree(dirname)
            os.makedirs(dirname)
            # self.eval_stub.ResetBestResponse(Empty())
            self.table.hand_history_location = 'hands/iteration_%d/eval_hand_%d/' % (self.iteration, self.ppo_eval_iterations_done+1)
            self.table.hand_history_enabled = True
            self.initial_obs = self.table.reset()
            self.table.hand_history_enabled = False
            if self.ppo_eval_iterations_done == N_EVAL_ITERATIONS:
                self.eval_workers_ready.set()
            else:
                self.ppo_last_training_info = int(2*N_PPO_TRAINING_HANDS/1000)
                self.ppo_last_eval_info = int(2*N_PPO_EVAL_HANDS/1000)
                self.ppo_training_workers_left = len(SLAVE_HOSTS)*N_PPO_EVAL_PROCESSES
                self.ppo_eval_workers_left = len(SLAVE_HOSTS) * N_PPO_EVAL_PROCESSES
                self.ppo_training_hands_left = N_PPO_TRAINING_HANDS
                self.ppo_evaluation_hands_left = N_PPO_EVAL_HANDS
        self.evaluation_lock.release()
        return Empty()

    def RequestMCTSEvaluation(self, request, context):
        self.evaluation_lock.acquire()
        retval = -1
        if self.mcts_hands_left > 0:
            self.mcts_hands_left -= 1
            # if self.evaluation_hands_left % 100 == 0:
            logging.info("%d evaluation hands remaining, current average exploitability %f bb/100" % (self.mcts_hands_left, 100*self.average_exploitability))
            retval = self.mcts_hands_left
        self.evaluation_lock.release()
        return IntMessage(value=retval)

    def ExitMCTSEvaluationPool(self, request, context):
        self.evaluation_lock.acquire()
        self.mcts_workers_left -= 1
        if self.mcts_workers_left == 0:
            self.eval_workers_ready.set()
        self.evaluation_lock.release()
        return Empty()

    def AddMCTSExploitabilitySample(self, request, context):
        self.evaluation_lock.acquire()
        self.total_eval_hands_played += 1
        self.average_exploitability = (self.total_eval_hands_played * self.average_exploitability + request.value) / (self.total_eval_hands_played + 1)
        self.evaluation_lock.release()
        return Empty()
