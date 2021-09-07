import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans, AgglomerativeClustering
from collections import Counter, defaultdict
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import json
from collections import defaultdict
import sys
import os
import torch

import pickle
import logging
import random
import pandas as pd
from argparse import ArgumentParser
import pdb
import sys

sys.path.append('./src')
from losses import *
from models import MLP, variable, MLP_adv
from process_data import load_data_deepmoji, upsample, load_data_biasbios

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)




SEED = 0

def set_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

def group_evaluation(preds, labels, p_labels, silence=True):

    preds = np.array(preds)
    labels = np.array(labels)
    p_labels = np.array(p_labels)

    p_set = set(p_labels)
    if len(p_set)!=2:
        print("Assuming binary private labels")

    g1_preds = preds[np.array(p_labels) == 1]
    g1_labels = labels[np.array(p_labels) == 1]

    g0_preds = preds[np.array(p_labels) == 0]
    g0_labels = labels[np.array(p_labels) == 0]

    tn0, fp0, fn0, tp0 = confusion_matrix(g0_labels, g0_preds).ravel()
    TPR0 = tp0/(fn0+tp0)
    TNR0 = tn0/(fp0+tn0)

    tn1, fp1, fn1, tp1 = confusion_matrix(g1_labels, g1_preds).ravel()
    TPR1 = tp1/(fn1+tp1)
    TNR1 = tn1/(tn1+fp1)
    
    f1_0 = f1_score(g0_labels, g0_preds, average='macro')
    f1_1 = f1_score(g1_labels, g1_preds, average='macro')

    if not silence:
        print("F1 0: {}".format(f1_0))
        print("F1 1: {}".format(f1_1))

        print("TPR 0: {}".format(TPR0))
        print("TPR 1: {}".format(TPR1))

        print("TNR 0: {}".format(TNR0))
        print("TNR 1: {}".format(TNR1))

        print("TPR gap: {}".format(  (TPR0-TPR1)))
        print("TNR gap: {}".format(abs(TNR0-TNR1)))

    return {"F1_0": f1_0,
            "F1_1": f1_1,
            "TPR_0":TPR0,
            "TPR_1":TPR1,
            "TNR_0":TNR0,
            "TNR_1":TNR1,
            "TPR_gap":abs(TPR0-TPR1),
            "TNR_gap":abs(TNR0-TNR1),
            "F1 GAP": abs(f1_0-f1_1)}

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
    print(X.shape, Y_p.shape, Y_m.shape)
    dataset = {'feature': X, 'labels': Y_m, 'protected_attribute': Y_p}
    return dataset

def run_loss_biasbios(option='biasbios', method='vanilla'):
    results = defaultdict(dict)
    DO_RANDOM = True
    DO_TUNE_LDAM = False
    DO_TUNE_FOCAL = False
    SAVE_CROSSENTROPY_300D = False
    
    logging.info('loading train dev test sets...')
    train_data = load_data_biasbios(os.path.join(args.datadir, 'train.pickle'), "./resources/professions.txt", os.path.join(args.datadir, "bert_encode_biasbios/train_cls.npy"))
    dev_data = load_data_biasbios(os.path.join(args.datadir, 'dev.pickle'), "./resources/professions.txt", os.path.join(args.datadir, "bert_encode_biasbios/dev_cls.npy"))
    test_data = load_data_biasbios(os.path.join(args.datadir, 'test.pickle'), "./resources/professions.txt", os.path.join(args.datadir, "bert_encode_biasbios/test_cls.npy"))
    x_train, y_p_train, y_m_train = train_data['feature'], train_data['protected_attribute'], train_data['labels']
    x_dev, y_p_dev, y_m_dev = dev_data['feature'], dev_data['protected_attribute'], dev_data['labels']
    x_test, y_p_test, y_m_test = test_data['feature'], test_data['protected_attribute'], test_data['labels']
    logging.info(f'train/dev/test data loaded. X_train: {x_train.shape} X_dev: {x_dev.shape} X_test: {x_test.shape}')

    if DO_RANDOM:
        model = DummyClassifier(strategy="most_frequent")
        model.fit(x_train, y_m_train)
        y_test_pred = model.predict(x_test)
        f1 = f1_score(y_m_test, y_test_pred, average='macro')
        _, biased_diffs = get_TPR(y_m_test, y_test_pred, y_p_test)
        results['bios']['rand'] = {"tpr": rms(list(biased_diffs.values()))}
        results['bios']['rand'].update({"f1": f1})
        #group_results = group_evaluation(y_test_pred, y_m_test, y_p_test)
        #results['rand'].update(group_results)


    unique, counts = np.unique(y_m_train, return_counts=True)
    cls_num_list = counts.tolist()
    beta = 0.9999
    #https://openaccess.thecvf.com/content_CVPR_2019/papers/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.pdf
    effective_num = 1.0 - np.power(beta, cls_num_list) 
    per_cls_weights = (1.0 - beta) / np.array(effective_num)
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
    per_cls_weights = variable(torch.FloatTensor(per_cls_weights))

    #protected classes effective weight (this is different from inverse frequency as it is way more smooth e.g. doesn't change the distro that much)
    unique, counts = np.unique(y_p_train, return_counts=True)
    clsp_num_list = counts.tolist()
    beta = 0.9999
    effective_num = 1.0 - np.power(beta, clsp_num_list)
    per_clsp_weights = (1.0 - beta) / np.array(effective_num)
    per_clsp_weights = per_clsp_weights / np.sum(per_clsp_weights) * len(clsp_num_list)
    per_clsp_weights = variable(torch.FloatTensor(per_clsp_weights))

    
    #instance_weights using effective number (smoothed version of inverse frequency)
    all_mps = []
    for m, p in zip(y_m_train, y_p_train):
        all_mps.append((m, p))
    all_mp_count = Counter(all_mps)
    pms = []
    pm_counts = []
    for k, v in all_mp_count.items():
        pms.append(k)
        pm_counts.append(v)
    pm_counts_id = {k:i for i, k in enumerate(pms)}
    y_mp_train = np.zeros_like(y_p_train)
    beta = 0.9999
    effective_num = 1.0 - np.power(beta, pm_counts)
    per_instance_weights = (1.0 - beta) / np.array(effective_num)
    per_instance_weights = per_instance_weights / np.sum(per_instance_weights) * len(pm_counts)
    instance_weights = np.zeros(y_p_train.shape[0])
    for i in range(instance_weights.shape[0]):
        instance_weights[i] = per_instance_weights[pm_counts_id[all_mps[i]]]
        y_mp_train[i] = pm_counts_id[all_mps[i]]


    _s = args.s #ldams
    _rho = args.rho #regul

    if method not in ['ldamadv', 'adv']:
        if method == 'vanilla':
            criterion = F.cross_entropy 
        elif method == 'iw':
            criterion = CrossEntropyWithInstanceWeights() #iw
        elif method == 'cw':
            criterion = nn.CrossEntropyLoss(weight=per_cls_weights) #cw
        elif method == 'focal':
            criterion = FocalLoss(weight=per_cls_weights, gamma=1)
        elif method == 'adjdice':
            criterion = SelfAdjDiceLoss()
        elif method == 'ldam':
            criterion = LDAMLoss(cls_num_list=cls_num_list, max_m=0.5, weight=None, s=_s) #ldam
        elif method == 'ldamreg':
            criterion = GeneralLDAMLoss(cls_num_list=cls_num_list, clsp_num_list=clsp_num_list, 
                mp_num_list=pm_counts, max_m=0.5, class_weight=None, ldams=_s, ldamc=1, 
                ldamg=0, ldamcg=0, use_instance=False, ldam_mul_c_g=False, rho=_rho)
        elif method == 'ldamaiw':
            criterion = GeneralLDAMLoss(cls_num_list=cls_num_list, clsp_num_list=clsp_num_list, 
                    mp_num_list=pm_counts, max_m=0.5, class_weight=None, ldams=_s, ldamc=1, 
                    ldamg=0, ldamcg=0, use_instance=True, ldam_mul_c_g=False) #ldamiw
        elif method == 'ldamcw':
            criterion = LDAMLoss(cls_num_list=cls_num_list, max_m=0.5, weight=per_cls_weights, s=_s) #ldam cw
        
        c_name = args.method
        set_seed(SEED)
        logging.info(f"loss: {c_name}")
        results[option][c_name] = {}
        dev_results[option][c_name] = {}

        normed_linear = True if 'ldam' in c_name.lower() else False
        model = MLP(input_size=x_train.shape[1], hidden_size=300, output_size=np.max(y_m_train) + 1, normed_linear=normed_linear, criterion=criterion)
        model.to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
        model.fit(x_train, y_m_train, y_p_train, y_mp_train, x_dev, y_m_dev, optimizer, instance_weights=instance_weights, n_iter=100, batch_size=128, max_patience=10)

        f1 = model.score(x_test, y_m_test)
        y_test_pred = model.predict(x_test)
        _, debiased_diffs = get_TPR(y_m_test, y_test_pred, y_p_test)
        results[option][c_name].update({"f1": f1})
        results[option][c_name].update({"tpr": rms(list(debiased_diffs.values()))})
        results[option][c_name].update({"acc": accuracy_score(y_m_test, y_test_pred)})
        group_results = group_evaluation(y_test_pred, y_m_test, y_p_test)
        results[option][c_name].update(group_results)

        dev_f1 = model.score(x_dev, y_m_dev)
        y_dev_pred = model.predict(x_dev)
        _, dev_debiased_diffs = get_TPR(y_m_dev, y_dev_pred, y_p_dev)
        dev_results[option][c_name].update({"f1": dev_f1})
        dev_results[option][c_name].update({"tpr": rms(list(dev_debiased_diffs.values()))})
        dev_results[option][c_name].update({"acc": accuracy_score(y_m_dev, y_dev_pred)})
        dev_group_results = group_evaluation(y_dev_pred, y_m_dev, y_p_dev)
        dev_results[option][c_name].update(dev_group_results)

    else: #adversarial methods
        adv_val = args.lamb
        logging.info(f"adv: {_s}_{adv_val}")
        normed_linear = True
        set_seed(SEED)
        adv_ldamcriterion = LDAMLoss(cls_num_list=cls_num_list, max_m=0.5, weight=None, s=_s) #ldam
        #advmodel = MLP_adv(input_size=x_train.shape[1], hidden_size=300, output_size=np.max(y_m_train) + 1, domain_output_size = np.max(y_p_train) + 1, normed_linear=normed_linear, criterion1=adv_ldamcriterion, criterion2=F.cross_entropy, alpha=adv_val)
        advmodel = MLP_adv(input_size=x_train.shape[1], hidden_size=300, output_size=np.max(y_m_train) + 1, domain_output_size = np.max(y_p_train) + 1, normed_linear=normed_linear, criterion1=adv_ldamcriterion, criterion2=F.cross_entropy, alpha=adv_val, rev_tech=1)
        advmodel.to(device)
        optimizer = torch.optim.Adam(params=advmodel.parameters(), lr=1e-3)
        advmodel.fit(x_train, y_m_train, y_p_train, y_mp_train, x_dev, y_m_dev, optimizer, instance_weights=instance_weights, n_iter=100, batch_size=128, max_patience=10)

        f1 = advmodel.score(x_test, y_m_test)
        y_test_pred = advmodel.predict(x_test)
        _, debiased_diffs = get_TPR(y_m_test, y_test_pred, y_p_test)
        _tpr_val  = rms(list(debiased_diffs.values()))
        results[option]["adv_{}_{}".format(_s, adv_val)] = {}
        results[option]["adv_{}_{}".format(_s, adv_val)].update({"f1": f1})
        results[option]["adv_{}_{}".format(_s, adv_val)].update({"tpr": _tpr_val})
        results[option]["adv_{}_{}".format(_s, adv_val)].update({"acc": accuracy_score(y_m_test, y_test_pred)})
        group_results = group_evaluation(y_test_pred, y_m_test, y_p_test)
        results[option]["adv_{}_{}".format(_s, adv_val)].update(group_results)

        logging.info(f"f1: {f1}")
        logging.info(f"tpr: {_tpr_val}")

        dev_f1 = advmodel.score(x_dev, y_m_dev)
        y_dev_pred = advmodel.predict(x_dev)
        _, dev_debiased_diffs = get_TPR(y_m_dev, y_dev_pred, y_p_dev)
        dev_results[option]["adv_{}_{}".format(_s, adv_val)] = {}
        dev_results[option]["adv_{}_{}".format(_s, adv_val)].update({"f1": dev_f1})
        dev_results[option]["adv_{}_{}".format(_s, adv_val)].update({"tpr": rms(list(dev_debiased_diffs.values()))})
        dev_results[option]["adv_{}_{}".format(_s, adv_val)].update({"acc": accuracy_score(y_m_dev, y_dev_pred)})
        dev_group_results = group_evaluation(y_dev_pred, y_m_dev, y_p_dev)
        dev_results[option]["adv_{}_{}".format(_s, adv_val)].update(dev_group_results)
            
    return results, dev_results 


def run_loss(train_data, dev_data, test_data, option='inlp0.5', method='vanilla'):
    results = defaultdict(dict)
    dev_results = defaultdict(dict)
    logging.info('loading train dev test sets...')
    x_train, y_p_train, y_m_train = train_data['feature'], train_data['protected_attribute'], train_data['labels']
    x_dev, y_p_dev, y_m_dev = dev_data['feature'], dev_data['protected_attribute'], dev_data['labels']
    x_test, y_p_test, y_m_test = test_data['feature'], test_data['protected_attribute'], test_data['labels']
    logging.info(f'train/dev/test data loaded. X_train: {x_train.shape} X_dev: {x_dev.shape} X_test: {x_test.shape}')

    unique, counts = np.unique(y_m_train, return_counts=True)
    cls_num_list = counts.tolist()
    beta = 0.9999
    #https://openaccess.thecvf.com/content_CVPR_2019/papers/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.pdf
    effective_num = 1.0 - np.power(beta, cls_num_list) 
    per_cls_weights = (1.0 - beta) / np.array(effective_num)
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
    per_cls_weights = variable(torch.FloatTensor(per_cls_weights))


    #protected classes effective weight (this is different from inverse frequency as it is way more smooth e.g. doesn't change the distro that much)
    unique, counts = np.unique(y_p_train, return_counts=True)
    clsp_num_list = counts.tolist()
    beta = 0.9999
    effective_num = 1.0 - np.power(beta, clsp_num_list)
    per_clsp_weights = (1.0 - beta) / np.array(effective_num)
    per_clsp_weights = per_clsp_weights / np.sum(per_clsp_weights) * len(clsp_num_list)
    per_clsp_weights = variable(torch.FloatTensor(per_clsp_weights))
    
    #instance_weights using effective number (smoothed version of inverse frequency)
    all_mps = []
    for m, p in zip(y_m_train, y_p_train):
        all_mps.append((m, p))
    all_mp_count = Counter(all_mps)
    pms = []
    pm_counts = []
    for k, v in all_mp_count.items():
        pms.append(k)
        pm_counts.append(v)
    pm_counts_id = {k:i for i, k in enumerate(pms)}
    y_mp_train = np.zeros_like(y_p_train)
    beta = 0.9999
    effective_num = 1.0 - np.power(beta, pm_counts)
    per_instance_weights = (1.0 - beta) / np.array(effective_num)
    per_instance_weights = per_instance_weights / np.sum(per_instance_weights) * len(pm_counts)
    instance_weights = np.zeros(y_p_train.shape[0])
    for i in range(instance_weights.shape[0]):
        instance_weights[i] = per_instance_weights[pm_counts_id[all_mps[i]]]
        y_mp_train[i] = pm_counts_id[all_mps[i]]

    
    _s = args.s #ldams
    _rho = args.rho #regul

    if method not in ['ldamadv', 'adv']:
        if method == 'vanilla':
            criterion = F.cross_entropy 
        elif method == 'iw':
            criterion = CrossEntropyWithInstanceWeights() #iw
        elif method == 'cw':
            criterion = nn.CrossEntropyLoss(weight=per_cls_weights) #cw
        elif method == 'focal':
            criterion = FocalLoss(weight=per_cls_weights, gamma=1)
        elif method == 'adjdice':
            criterion = SelfAdjDiceLoss()
        elif method == 'ldam':
            criterion = LDAMLoss(cls_num_list=cls_num_list, max_m=0.5, weight=None, s=_s) #ldam
        elif method == 'ldamreg':
            criterion = GeneralLDAMLoss(cls_num_list=cls_num_list, clsp_num_list=clsp_num_list, 
                mp_num_list=pm_counts, max_m=0.5, class_weight=None, ldams=_s, ldamc=1, 
                ldamg=0, ldamcg=0, use_instance=False, ldam_mul_c_g=False, rho=_rho)
        elif method == 'ldamaiw':
            criterion = GeneralLDAMLoss(cls_num_list=cls_num_list, clsp_num_list=clsp_num_list, 
                    mp_num_list=pm_counts, max_m=0.5, class_weight=None, ldams=_s, ldamc=1, 
                    ldamg=0, ldamcg=0, use_instance=True, ldam_mul_c_g=False) #ldamiw
        elif method == 'ldamcw':
            criterion = LDAMLoss(cls_num_list=cls_num_list, max_m=0.5, weight=per_cls_weights, s=_s) #ldam cw
        
        c_name = args.method
        set_seed(SEED)
        logging.info(f"loss: {c_name}")
        results[option][c_name] = {}
        dev_results[option][c_name] = {}

        normed_linear = True if 'ldam' in c_name.lower() else False
        model = MLP(input_size=x_train.shape[1], hidden_size=300, output_size=np.max(y_m_train) + 1, normed_linear=normed_linear, criterion=criterion)
        model.to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
        model.fit(x_train, y_m_train, y_p_train, y_mp_train, x_dev, y_m_dev, optimizer, instance_weights=instance_weights, n_iter=100, batch_size=128, max_patience=10)

        f1 = model.score(x_test, y_m_test)
        y_test_pred = model.predict(x_test)
        _, debiased_diffs = get_TPR(y_m_test, y_test_pred, y_p_test)
        results[option][c_name].update({"f1": f1})
        results[option][c_name].update({"tpr": rms(list(debiased_diffs.values()))})
        results[option][c_name].update({"acc": accuracy_score(y_m_test, y_test_pred)})
        group_results = group_evaluation(y_test_pred, y_m_test, y_p_test)
        results[option][c_name].update(group_results)

        dev_f1 = model.score(x_dev, y_m_dev)
        y_dev_pred = model.predict(x_dev)
        _, dev_debiased_diffs = get_TPR(y_m_dev, y_dev_pred, y_p_dev)
        dev_results[option][c_name].update({"f1": dev_f1})
        dev_results[option][c_name].update({"tpr": rms(list(dev_debiased_diffs.values()))})
        dev_results[option][c_name].update({"acc": accuracy_score(y_m_dev, y_dev_pred)})
        dev_group_results = group_evaluation(y_dev_pred, y_m_dev, y_p_dev)
        dev_results[option][c_name].update(dev_group_results)

    else: #adversarial methods
        adv_val = args.lamb
        logging.info(f"adv: {_s}_{adv_val}")
        normed_linear = True
        set_seed(SEED)
        adv_ldamcriterion = LDAMLoss(cls_num_list=cls_num_list, max_m=0.5, weight=None, s=_s) #ldam
        #advmodel = MLP_adv(input_size=x_train.shape[1], hidden_size=300, output_size=np.max(y_m_train) + 1, domain_output_size = np.max(y_p_train) + 1, normed_linear=normed_linear, criterion1=adv_ldamcriterion, criterion2=F.cross_entropy, alpha=adv_val)
        advmodel = MLP_adv(input_size=x_train.shape[1], hidden_size=300, output_size=np.max(y_m_train) + 1, domain_output_size = np.max(y_p_train) + 1, normed_linear=normed_linear, criterion1=adv_ldamcriterion, criterion2=F.cross_entropy, alpha=adv_val, rev_tech=1)
        advmodel.to(device)
        optimizer = torch.optim.Adam(params=advmodel.parameters(), lr=1e-3)
        advmodel.fit(x_train, y_m_train, y_p_train, y_mp_train, x_dev, y_m_dev, optimizer, instance_weights=instance_weights, n_iter=100, batch_size=128, max_patience=10)

        f1 = advmodel.score(x_test, y_m_test)
        y_test_pred = advmodel.predict(x_test)
        _, debiased_diffs = get_TPR(y_m_test, y_test_pred, y_p_test)
        _tpr_val  = rms(list(debiased_diffs.values()))
        results[option]["adv_{}_{}".format(_s, adv_val)] = {}
        results[option]["adv_{}_{}".format(_s, adv_val)].update({"f1": f1})
        results[option]["adv_{}_{}".format(_s, adv_val)].update({"tpr": _tpr_val})
        results[option]["adv_{}_{}".format(_s, adv_val)].update({"acc": accuracy_score(y_m_test, y_test_pred)})
        group_results = group_evaluation(y_test_pred, y_m_test, y_p_test)
        results[option]["adv_{}_{}".format(_s, adv_val)].update(group_results)

        logging.info(f"f1: {f1}")
        logging.info(f"tpr: {_tpr_val}")

        dev_f1 = advmodel.score(x_dev, y_m_dev)
        y_dev_pred = advmodel.predict(x_dev)
        _, dev_debiased_diffs = get_TPR(y_m_dev, y_dev_pred, y_p_dev)
        dev_results[option]["adv_{}_{}".format(_s, adv_val)] = {}
        dev_results[option]["adv_{}_{}".format(_s, adv_val)].update({"f1": dev_f1})
        dev_results[option]["adv_{}_{}".format(_s, adv_val)].update({"tpr": rms(list(dev_debiased_diffs.values()))})
        dev_results[option]["adv_{}_{}".format(_s, adv_val)].update({"acc": accuracy_score(y_m_dev, y_dev_pred)})
        dev_group_results = group_evaluation(y_dev_pred, y_m_dev, y_p_dev)
        dev_results[option]["adv_{}_{}".format(_s, adv_val)].update(dev_group_results)
            
    return results, dev_results

def pretty_print(results, option='original', output_csv_dir='./', split = "test"):

    with open(f"{option}_results_INLP_ACL_Table1_wacc_{split}_seed_{SEED}.pkl", "wb") as f:
        pickle.dump(results, f)

    for option, res in results.items():
        df = pd.DataFrame(res)
        df.to_csv(os.path.join(output_csv_dir, f"{option}_results_results_INLP_ACL_Table1_wacc_{split}_seed_{SEED}.csv"))


def pretty_print_biography(results, option='original', output_csv_dir='./', split = "test"):

    with open(f"{option}_bios_twoclass_results_april19_large_{split}_seed_{SEED}.pkl", "wb") as f:
        pickle.dump(results, f)
    for option, res in results.items():
        df = pd.DataFrame(res)
        df.to_csv(os.path.join(output_csv_dir, f"{option}_bios_twoclass_results_april19_large_{split}_seed_{SEED}.csv"))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-dataset", type=str, default="deepmoji", help="deepmoji or biography")
    #parser.add_argument("-ratio", type=float, default=0.5, choices=[0.5, 0.6, 0.7, 0.8], help="sterotype ratio: table 1 in paper")
    parser.add_argument("-datadir", type=str, default="", help="the directory where deepmoji is located.", required=True)
    parser.add_argument("-rho", type=float, default=0, help="rho parameter in fairness regularisation")
    parser.add_argument("-method", type=str, choices=['vanilla', 'ldamcw', 'ldamiw', 'ldamreg', 'ldamadv', 'reg', 'adv'])
    parser.add_argument("-s", type=float, help="C parameter in ldam")
    parser.add_argument("-lamb", type=float, help="lambda parameter in adversarial method")



    args = parser.parse_args()
    ds = args.dataset
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"using device {device}")


    #table 1
    if ds == 'deepmoji':
        ratio = float(args.emojidir.split('_')[-1])
        traindata = load_data(f'{args.datadir}/train/', size=100000, ratio=ratio)
        devdata = load_data(f'{args.datadir}/dev/', size=100000, ratio=ratio)
        testdata = load_data(f'{args.datadir}/test/', size=100000, ratio=0.5)
        test_results = defaultdict(dict)
        dev_results = defaultdict(dict)
        test_run, dev_run = run_loss(traindata, devdata, testdata, option='inlp_{}'.format(ratio), method=args.method)
        test_results.update(test_run)
        dev_results.update(dev_run)
        pretty_print(test_results, option='inlp_{}'.format(ratio))
        pretty_print(dev_results, option='inlp_{}'.format(ratio), split = "dev")
    elif ds == "biography":
        test_results = defaultdict(dict)
        dev_results = defaultdict(dict)
        test_run, dev_run = run_loss_biasbios(method=args.method)
        test_results.update(test_run)
        dev_results.update(dev_run)
        pretty_print_biography(test_results)
        pretty_print_biography(dev_results, split = "dev")
            
            
        