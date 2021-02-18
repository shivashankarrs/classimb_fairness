import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle

from collections import Counter, defaultdict

import json
from collections import defaultdict
import sys
import os
import torch
from losses import *
from models import MLP, LogReg, variable
import pickle

def load_data(path, size, ratio=0.5):
    fnames = ["neg_neg.npy", "neg_pos.npy", "pos_neg.npy", "pos_pos.npy"]
    protected_labels = [0, 1, 0, 1]
    main_labels = [0, 0, 1, 1]
    X, Y_p, Y_m = [], [], []
    n1 = int(size * ratio / 2)
    n2 = int(size * (1 - ratio) / 2)
#     print(n1, n2)

    for fname, p_label, m_label, n in zip(fnames, protected_labels, main_labels, [n1, n2, n2, n1]):
#         print(path + '/' + fname)
#         print(np.load(path + '/' + fname).shape)
        data = np.load(path + '/' + fname)[:n]
        for x in data:
            X.append(x)
        for _ in data:
            Y_p.append(p_label)
        for _ in data:
            Y_m.append(m_label)

    Y_p = np.array(Y_p)
    Y_m = np.array(Y_m)
    X = np.array(X)
    X, Y_p, Y_m = shuffle(X, Y_p, Y_m, random_state=0)
    return X, Y_p, Y_m


def get_TPR(y_main, y_hat_main, y_protected):
    
    all_y = list(Counter(y_main).keys())
    
    protected_vals = defaultdict(dict)
    for label in all_y:
        for i in range(2):
            used_vals = (y_main == label) & (y_protected == i)
            y_label = y_main[used_vals]
            y_hat_label = y_hat_main[used_vals]
            protected_vals['y:{}'.format(label)]['p:{}'.format(i)] = (y_label == y_hat_label).mean()
            
    diffs = {}
    for k, v in protected_vals.items():
        vals = list(v.values())
        diffs[k] = vals[0] - vals[1]
    return protected_vals, diffs




def rms(arr):
    return np.sqrt(np.mean(np.square(arr)))



def run_all_losses(ratio=0.5):
    results = defaultdict(dict)
    x_train, y_p_train, y_m_train = load_data(
        '../datasets/emoji_sent_race_{}/train/'.format(ratio),
        size=100000, ratio=ratio)
    x_dev, y_p_dev, y_m_dev = load_data(
        '../datasets/emoji_sent_race_{}/test/'.format(ratio),
        size=100000, ratio=0.5)
    x_dev_realdev, y_p_dev_realdev, y_m_dev_realdev = load_data(
        '../datasets/emoji_sent_race_{}/dev/'.format(ratio),
        size=100000, ratio=0.5)
    
    biased_classifier = LinearSVC(fit_intercept=True, class_weight='balanced', dual=False, C=0.1, max_iter=10000)

    biased_classifier.fit(x_train, y_m_train)
    biased_score = biased_classifier.score(x_dev, y_m_dev)
    
    '''
    P = np.load('../data/emoji_sent_race_{}/P_svm.num-clfs=300.npy'.format(ratio), allow_pickle=True)
    P = P[1]
    n_dims = 120
#     n_dims = 70
    if ratio == 0.5:
        n_dims = 200
    elif ratio == 0.6:
        n_dims = 100
    elif ratio == 0.7:
        n_dims = 115
    elif ratio == 0.8:
        n_dims = 200
    P = debias.get_projection_to_intersection_of_nullspaces(P[:n_dims], input_dim=300)
    
    debiased_x_train = P.dot(x_train.T).T
    debiased_x_dev = P.dot(x_dev.T).T

    classifier = LinearSVC(fit_intercept=True, class_weight='balanced', dual=False, C=0.1, max_iter=10000)

    classifier.fit(debiased_x_train, y_m_train)
    debiased_score = classifier.score(debiased_x_dev, y_m_dev)
    '''
    #we are not predicting protected attributes from the debiased representations, we are predicting from the original attributes, so the results are not important
    p_classifier = SGDClassifier(warm_start=True, loss='log', n_jobs=64, max_iter=10000, random_state=0, tol=1e-3)
    p_classifier.fit(x_train, y_p_train)
    p_score = p_classifier.score(x_dev, y_p_dev)
    #results[ratio]['p_acc'] = p_score
    _, biased_diffs = get_TPR(y_m_dev, biased_classifier.predict(x_dev), y_p_dev)
    
   # _, debiased_diffs = get_TPR(y_m_dev, classifier.predict(debiased_x_dev), y_p_dev)
    
#     results[ratio]['biased_diff_tpr'] = biased_diffs['y:0']
    results[ratio]['sgd_tpr'] = rms(list(biased_diffs.values()))
#     results[ratio]['debiased_diff_tpr'] = debiased_diffs['y:0']
    #results[ratio]['debiased_diff_tpr'] = rms(list(debiased_diffs.values()))
    
    results[ratio]['sgd_acc'] = biased_score
    #results[ratio]['debiased_acc'] = debiased_score
    
    #added by afshin
    unique, counts = np.unique(y_m_train, return_counts=True)
    cls_num_list = counts.tolist()
    beta = 0.9999
    effective_num = 1.0 - np.power(beta, cls_num_list)
    per_cls_weights = (1.0 - beta) / np.array(effective_num)
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
    per_cls_weights = variable(torch.FloatTensor(per_cls_weights))
    
    criterion1 = FocalLoss(weight=per_cls_weights, gamma=1)
    criterion2 = LDAMLoss(cls_num_list=cls_num_list, max_m=0.5, weight=None, s=30)
    criterion3 = SelfAdjDiceLoss()
    criterion4 = F.cross_entropy
    criterions = {'ldam':criterion2, 'focal':criterion1, 'adjdice':criterion3, 'crosent':criterion4}
    for c_name, criterion in criterions.items():
        debiased_model = MLP(input_size=x_train.shape[1], hidden_size=300, output_size=np.max(y_m_train) + 1, normed_linear=True, criterion=criterion2)
        #debiased_model = LogReg(input_size=x_train.shape[1], output_size=np.max(y_m_train) + 1, 
        #                        normed_linear=True if c_name=='ldam' else False, criterion=criterion)
        debiased_model.to(device)
        optimizer = torch.optim.Adam(params=debiased_model.parameters(), lr=0.001)
        debiased_model.fit(x_train, y_m_train, x_dev_realdev, y_m_dev_realdev, optimizer, n_iter=200, batch_size=1000, max_patience=10)
        #get the representation from the trained MLP
        if isinstance(debiased_model, MLP):
            x_train_repr, x_dev_realdev_repr, x_dev_repr = debiased_model.get_hidden(x_train), debiased_model.get_hidden(x_dev_realdev), debiased_model.get_hidden(x_dev)
            out_filename = '../datasets/emoji_sent_race_{}/x_{}_reps.pkl'.format(ratio, c_name)
            with open(out_filename, 'wb') as fout:
                print('dumping repr in ' + out_filename)
                pickle.dump((x_train_repr, x_dev_realdev_repr, x_dev_repr), fout)
        debiased_score = debiased_model.score(x_dev, y_m_dev)
        _, debiased_diffs = get_TPR(y_m_dev, debiased_model.predict(x_dev), y_p_dev)
        results[ratio][f"{c_name}_acc"] = debiased_score
        results[ratio][f"{c_name}_tpr"] = rms(list(debiased_diffs.values()))
    pretty_print(results, ratio)
    return results    


def pretty_print(results, ratio=0.5):
    accs = [h for h in sorted(results[ratio]) if 'acc' in h]
    tprs = [h for h in sorted(results[ratio]) if 'tpr' in h]
    header = "ratio" + " " + " ".join(accs) + " " +  " ".join(tprs)
    print(header)

    for r, v in results.items():
        row = str(r) + " " + " ".join(['{:.2f}'.format(v[a]) for a in accs]) + " " + " ".join(['{:.2f}'.format(v[a]) for a in tprs]) 
        print(row)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_results = defaultdict(dict)
    for ratio in [0.5, 0.6, 0.7, 0.8]:
        ratio_results = run_all_losses(ratio)
        all_results[ratio] = ratio_results[ratio]
    pretty_print(all_results)


