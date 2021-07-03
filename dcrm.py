import logging
import argparse
import grpc
from client.client import deep_cfr

modes = ['client', 'server', 'dual']

if __name__ == "__main__":
    options = [('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)]
    channel = grpc.insecure_channel('localhost:50051', options)
    stub_learner = LearnerStub(channel)
