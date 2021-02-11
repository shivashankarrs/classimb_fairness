import pickle

def load_dataset(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def dump_dataset(path, dev_extend):
    with open(path, "wb") as f:
        data = pickle.dump(dev_extend, f)
    return 

def process():

    return

def get_data():
    
    return

