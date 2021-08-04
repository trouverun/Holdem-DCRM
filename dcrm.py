import os
import logging
import argparse
import grpc
from rpc import RL_pb2_grpc
from concurrent.futures import ThreadPoolExecutor
from server.server import serve
from masterslave import Master, Slave
from config import MASTER_HOST, SLAVE_HOSTS

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', help='master/slave/server', type=str)
    parser.add_argument('-host', help='host address used for the slave machine', type=str)
    parser.add_argument('-hosts', help='host address(es) used for the actor-/regret server(s)', type=str, nargs='+')
    args = parser.parse_args()
    options = [('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)]

    if args.mode == 'master':
        server = grpc.server(ThreadPoolExecutor(max_workers=2), options=options)
        RL_pb2_grpc.add_MasterServicer_to_server(Master(), server)
        server.add_insecure_port(MASTER_HOST)
        server.start()
        logging.info("Master server serving at %s", MASTER_HOST)
        server.wait_for_termination()
    elif args.mode == 'slave':
        if args.host not in SLAVE_HOSTS:
            raise ValueError("Unrecognized hostname for slave, check the config file.")

        dirs = os.listdir()
        if 'hands' not in dirs:
            os.makedirs('hands')

        server = grpc.server(ThreadPoolExecutor(max_workers=2), options=options)
        RL_pb2_grpc.add_SlaveServicer_to_server(Slave(), server)
        server.add_insecure_port(args.host)
        server.start()
        logging.info("Slave server serving at %s", args.hosts[0])
        server.wait_for_termination()
    elif args.mode == 'server':
        serve(args.hosts)


