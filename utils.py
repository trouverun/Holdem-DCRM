import pickle

def fast_deep_copy(object):
    bytes = pickle.dumps(object)
    return pickle.loads(bytes)