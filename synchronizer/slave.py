import logging
from multiprocessing import Process, Queue
from rpc.RL_pb2 import Empty
from rpc.RL_pb2_grpc import SlaveServicer
from config import N_PLAYERS, N_QUE_PROCESS, N_CONC_TRAVERSALS_PER_PROCESS, N_TRAVERSE_PROCESSES, CLIENT_SAMPLES_MIN_BATCH_SIZE
from client.client import clear_queue_process, traverse_process
from threading import Thread


class Slave(SlaveServicer):
    def __init__(self):
        self.identifier = None
        self.traverser_que = Queue()
        self.que_processes = []
        Process(target=self._traverse_background_thread, args=()).start()

    def cleanup(self):
        for p in self.que_processes:
            p.terminate()
        logging.info("slave cleanup done")

    def _traverse_background_thread(self):
        regret_ques = [Queue(maxsize=5*CLIENT_SAMPLES_MIN_BATCH_SIZE) for _ in range(N_PLAYERS)]
        strategy_ques = [Queue(maxsize=5*CLIENT_SAMPLES_MIN_BATCH_SIZE) for _ in range(N_PLAYERS)]
        self.que_processes = [Process(target=clear_queue_process, args=(regret_ques, strategy_ques)) for _ in range(N_QUE_PROCESS)]
        for p in self.que_processes:
            p.start()
        while True:
            traverser = self.traverser_que.get()
            process_count = 0
            self.traverse_processes = [
                Process(target=traverse_process, args=(N_CONC_TRAVERSALS_PER_PROCESS, traverser, regret_ques[traverser], strategy_ques))
                for n in range(N_TRAVERSE_PROCESSES)
            ]
            for p in self.traverse_processes:
                process_count += 1
                logging.info("starting traversal process %d" % process_count)
                p.start()
            for p in self.traverse_processes:
                p.join()
                process_count -= 1
                logging.info("joined traversal process %d" % process_count)

    def SetIdentifier(self, request, context):
        self.identifier = request.value
        return Empty()

    def RunTraversals(self, request, context):
        self.traverser_que.put(request.value)
        return Empty()