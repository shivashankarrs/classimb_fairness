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
import logging
import sys
sys.path.append('../')

from data.process_data import load_data_deepmoji

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

def load_data(path):
    fnames = ["neg_neg.npy", "neg_pos.npy", "pos_neg.npy", "pos_pos.npy"]
    protected_labels = [0, 1, 0, 1]
    main_labels = [0, 0, 1, 1]
    X, Y_p, Y_m = [], [], []

#     print(n1, n2)

    for fname, p_label, m_label in zip(fnames, protected_labels, main_labels):
#         print(path + '/' + fname)
#         print(np.load(path + '/' + fname).shape)
        data = np.load(path + '/' + fname)
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



def run_all_losses(option='original'):
    ratio = option

    results = defaultdict(dict)
    logging.info('loading train dev test sets...')
    train_data = load_data_deepmoji('../datasets/deepmoji/train', option=option)
    dev_data = load_data_deepmoji('../datasets/deepmoji/dev', option=option)
    test_data = load_data_deepmoji('../datasets/deepmoji/test', option=option)
    x_train, y_p_train, y_m_train = train_data['feature'], train_data['protected_attribute'], train_data['labels']
    x_dev, y_p_dev, y_m_dev = dev_data['feature'], dev_data['protected_attribute'], dev_data['labels']
    x_test, y_p_test, y_m_test = test_data['feature'], test_data['protected_attribute'], test_data['labels']

    logging.info(f'train/dev/test data loaded. X_train: {x_train.shape} X_dev: {x_dev.shape} X_test: {x_test.shape}')
    biased_classifier = LinearSVC(fit_intercept=True, class_weight='balanced', dual=False, C=0.1, max_iter=10000)

    biased_classifier.fit(x_train, y_m_train)
    biased_score = biased_classifier.score(x_test, y_m_test)
    
    #we are not predicting protected attributes from the debiased representations, we are predicting from the original attributes, so the results are not important
    p_classifier = SGDClassifier(warm_start=True, loss='log', n_jobs=64, max_iter=10000, random_state=0, tol=1e-3)
    p_classifier.fit(x_train, y_p_train)
    p_score = p_classifier.score(x_test, y_p_test)
    _, biased_diffs = get_TPR(y_m_test, biased_classifier.predict(x_test), y_p_test)
    results[option]['sgd_tpr'] = rms(list(biased_diffs.values()))
    results[option]['sgd_acc'] = biased_score

    
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
        logging.info(f"loss: {c_name}")
        #only for ldam the last layer weights should be normalised so we'll have a normed_linear layer
        normed_linear = True if c_name == 'ldam' else False
        debiased_model = MLP(input_size=x_train.shape[1], hidden_size=300, output_size=np.max(y_m_train) + 1, normed_linear=normed_linear, criterion=criterion)
        #debiased_model = LogReg(input_size=x_train.shape[1], output_size=np.max(y_m_train) + 1, 
        #                        normed_linear=True if c_name=='ldam' else False, criterion=criterion)
        debiased_model.to(device)
        optimizer = torch.optim.Adam(params=debiased_model.parameters(), lr=1e-3)
        debiased_model.fit(x_train, y_m_train, x_dev, y_m_dev, optimizer, n_iter=1000, batch_size=1000, max_patience=20)
        #get the representation from the trained MLP
        if isinstance(debiased_model, MLP) and criterion == criterion4:
            x_train_repr, x_dev_repr, x_test_repr = debiased_model.get_hidden(x_train), debiased_model.get_hidden(x_dev), debiased_model.get_hidden(x_test)
            np.save('../datasets/deepmoji/x_{}_300d.npy'.format('train'), x_train_repr)
            np.save('../datasets/deepmoji/x_{}_300d.npy'.format('dev'), x_dev_repr)
            np.save('../datasets/deepmoji/x_{}_300d.npy'.format('test'), x_test_repr)

        debiased_score = debiased_model.score(x_test, y_m_test)
        _, debiased_diffs = get_TPR(y_m_test, debiased_model.predict(x_test), y_p_test)
        results[option][f"{c_name}_acc"] = debiased_score
        results[option][f"{c_name}_tpr"] = rms(list(debiased_diffs.values()))
    pretty_print(results, option)
    return results    


def pretty_print(results, option='original'):
    accs = [h for h in sorted(results[option]) if 'acc' in h]
    tprs = [h for h in sorted(results[option]) if 'tpr' in h]
    header = "option" + " " + " ".join(accs) + " " +  " ".join(tprs)
    print(header)

    for r, v in results.items():
        row = str(r) + " " + " ".join(['{:.2f}'.format(v[a]) for a in accs]) + " " + " ".join(['{:.2f}'.format(v[a]) for a in tprs]) 
        print(row)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_results = defaultdict(dict)
    for option in ['original']:
        option_results = run_all_losses(option=option)
        all_results[option] = option_results[option]
    pretty_print(all_results)


