import pickle
import numpy as np
from sklearn.utils import shuffle

def load_data_deepmoji(path):
    fnames = ["neg_neg.npy", "neg_pos.npy", "pos_neg.npy", "pos_pos.npy"]
    protected_labels = [0, 1, 0, 1]
    main_labels = [0, 0, 1, 1]
    dataset = []
    for fname, p_label, m_label in zip(fnames, protected_labels, main_labels):
        print(path + '/' + fname)
        data = np.load(path + '/' + fname)
        for x in data:
            _data = {}
            _data['features'] = x 
            _data['protected_feature'] = p_label
            _data['label'] = m_label
            dataset.append(_data)
    dataset = shuffle(dataset, random_state=0)
    return dataset

def load_dictionary(path):
    with open(path, "r", encoding = "utf-8") as f:
        lines = f.readlines()
    text2index = {}
    for line in lines:
        k,v = line.strip().split("\t")
        v = int(v)
        text2index[k] = v
    return text2index

def load_data_biasbios(path, rpath):
    fulldata = load_dataset(path)
    prof2index = load_dictionary(rpath)
    trimdataset = []
    for data in fulldata:
        _data = {}
        _data['text'] = data['hard_text']
        _data['protected_feature'] = 1 if data['g'] == 'f' else 0
        _data['label'] = prof2index[data['p']]
        trimdataset.append(_data)
    trimdataset = shuffle(trimdataset, random_state=0)
    return trimdataset

def load_dataset(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def dump_dataset(path, data):
    with open(path, "wb") as f:
        data = pickle.dump(data, f)
    return 

