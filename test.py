import numpy as np
import time
from multiprocessing import Queue


def step_1(que):
    data1 = []
    data2 = []
    data3 = []
    data4 = []

    for i in range(int(1e4)):
        data1.append(np.zeros([5, 60]))
        data2.append(np.zeros(1))
        data3.append(np.zeros(4))
        data4.append(np.zeros(7))

    que.put((data1, data2, data3, data4))


def step_2(que):
    for i in range(int(1e5)):
        que.put((np.zeros([5, 60]), np.zeros(1), np.zeros(4), np.zeros(7)))


def process_data_1(que):
    data1 = []
    data2 = []
    data3 = []
    data4 = []

    while True:
        d1, d2, d3, d4 = que.get()
        if d1 is None:
            break
        data1.append(d1)
        data2.append(d2)
        data3.append(d3)
        data4.append(d4)


def process_data_2(que):
    data1 = []
    data2 = []
    data3 = []
    data4 = []

    while True:
        d1, d2, d3, d4 = que.get()
        if d1 is None:
            break
        data1.extend(d1)
        data2.extend(d2)
        data3.extend(d3)
        data4.extend(d4)


que = Queue()

start1 = time.time()
step_1(que)
que.put((None, None, None, None))
process_data_1(que)
end1 = time.time()

start2 = time.time()
step_2(que)
que.put((None, None, None, None))
process_data_2(que)
end2 = time.time()

print(end1-start1)
print(end2-start2)

for player in range(N_PLAYERS):
    iter_que.put(('mode', 'traversal'))
    traverse_processes = [
        Process(target=traverse_process, args=(traverse_channels[n], N_CONC_TRAVERSALS_PER_PROCESS, loops_per_process, player,
                                               regret_ques[player], strategy_ques, iter_que))
        for n in range(N_TRAVERSE_PROCESSES)
    ]
    for p in traverse_processes:
        process_count += 1
        logging.debug("starting traversal process %d" % process_count)
        p.start()
    for p in traverse_processes:
        p.join()
        process_count -= 1
        logging.debug("joined traversal process %d" % process_count)
    logging.info("Training regrets for player %d" % player)
    stub_learner.TrainRegrets(IntMessage(value=player))
    iter_que.put(('reset', None))
logging.info("Training strategy for iteration %d" % iteration)
stub_learner.TrainStrategy(Empty())