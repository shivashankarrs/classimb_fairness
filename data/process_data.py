import pickle
import numpy as np
import os, math


DATA_PATH = "/Users/sssub/Documents/GitHub/classimb_fairness/class_imbalance/datasets/"

def get_sizes(base_size):
    black = 0.18 #8780
    white = 0.82 #40000
    pos = int(base_size + (base_size*0.18/0.82))
    neg = int(pos * 0.31/0.69)
    print(base_size, pos, neg)
    pos_pos_size, pos_neg_size, neg_pos_size, neg_neg_size = [math.ceil(pos*black), math.ceil(pos*white), math.ceil(neg*black), \
                                                          math.ceil(neg*white)]
    print(pos_pos_size, pos_neg_size, neg_pos_size, neg_neg_size)
    sample_size = [pos_pos_size, pos_neg_size, neg_pos_size, neg_neg_size]
    return sample_size

def get_inlp_sizes(base_size, black_ratio, white_ratio):
    black = black_ratio
    white = white_ratio
    pos = base_size
    neg = base_size
    pos_pos_size, pos_neg_size, neg_pos_size, neg_neg_size = [math.ceil(pos*black), math.ceil(pos*white), math.ceil(neg*white), \
                                                          math.ceil(neg*black)]
    print(pos_pos_size, pos_neg_size, neg_pos_size, neg_neg_size)
    sample_size = [pos_pos_size, pos_neg_size, neg_pos_size, neg_neg_size]
    return sample_size

def downsample(vecs, size):
    np.random.seed(1)
    np.random.shuffle(vecs)
    return vecs[:size]

def load_data_deepmoji(path, option='original'):
    fnames = ["pos_pos.npy", "pos_neg.npy", "neg_pos.npy", "neg_neg.npy"]
    protected_labels = [1, 0, 1, 0]
    main_labels = [1, 1, 0, 0]
    xdata = []
    labels = []
    protected_attribute = []

    check_data = np.load(path + '/' + "neg_neg.npy")
    
    if option=='original':
        sample_size = get_sizes(check_data.shape[0])
    elif option=='inlp0.5':
        sample_size = get_inlp_sizes(check_data.shape[0], 0.5, 0.5)
    elif option=='inlp0.6':
        sample_size = get_inlp_sizes(check_data.shape[0], 0.6, 0.4)
    elif option=='inlp0.7':
        sample_size = get_inlp_sizes(check_data.shape[0], 0.7, 0.3)
    elif option=='inlp0.8':
        sample_size = get_inlp_sizes(check_data.shape[0], 0.8, 0.2)
        
    print(check_data.shape[0], sample_size)

    for fname, p_label, m_label, size in zip(fnames, protected_labels, main_labels, sample_size):
        print(path + '/' + fname)
        data = np.load(path + '/' + fname)
        sampled_data = downsample(data, size)
        for x in sampled_data:
            xdata.append(x)
            protected_attribute.append(p_label)
            labels.append(m_label)
        print(fname, sampled_data.shape, len(xdata), len(protected_attribute), len(labels))
    dataset = {'feature': np.array(xdata), 'labels': np.array(labels), 'protected_attribute': np.array(protected_attribute)}
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

def load_data_biasbios(path, rpath, xpath):
    fdata = load_dataset(path)
    prof2index = load_dictionary(rpath)
    xdata = np.load(xpath)
    labels = []
    protected_attribute = []
    for data in fdata:
        _protected_feature = 1 if data['g'] == 'f' else 0
        _label = prof2index[data['p']]
        labels.append(_label)
        protected_attribute.append(_protected_feature)
    print(xdata.shape, len(protected_attribute), len(labels))
    dataset = {'feature': np.array(xdata), 'labels': np.array(labels), 'protected_attribute': np.array(protected_attribute)}
    return dataset

def load_dataset(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def dump_dataset(path, data):
    with open(path, "wb") as f:
        data = pickle.dump(data, f)
    return 

if __name__ == "__main__":
    train_biasbios = load_data_biasbios(DATA_PATH+"biography/test.pickle", "../resources/professions.txt", DATA_PATH+"biography/features/bert_encode_biasbios/test_cls.npy")
    print (len(train_biasbios), train_biasbios['feature'].shape, train_biasbios['feature'][0].shape)

    #train_biasbios = load_data_deepmoji(DATA_PATH+"deepmoji/test/", option='inlp0.8')
    #from collections import Counter
    #print (len(train_biasbios), len(train_biasbios['labels']), train_biasbios['feature'].shape)
    #print (Counter(train_biasbios['labels']), Counter(train_biasbios['protected_attribute']))