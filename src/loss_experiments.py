import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from collections import Counter, defaultdict
import json
from collections import defaultdict
import sys
import os
import torch
from losses import *
from models import MLP, variable
import pickle
import logging
import pandas as pd
import pdb
import sys
sys.path.append('../')

from data.process_data import load_data_deepmoji

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

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

        print("TPR gap: {}".format(abs(TPR0-TPR1)))
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


def tune_ldam(x_train, y_m_train, x_dev, y_m_dev):
    unique, counts = np.unique(y_m_train, return_counts=True)
    cls_num_list = counts.tolist()
    beta = 0.9999
    effective_num = 1.0 - np.power(beta, cls_num_list)
    per_cls_weights = (1.0 - beta) / np.array(effective_num)
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
    per_cls_weights = variable(torch.FloatTensor(per_cls_weights))
    results = defaultdict(dict)
    best_f1, best_s, best_max_m = 0, 0, 0
    for s in [1, 4, 8, 16, 32, 64]:
        for max_m in [0.5]:
            criterion = LDAMLoss(cls_num_list=cls_num_list, max_m=max_m, weight=per_cls_weights, s=s)
            #only for ldam the last layer weights should be normalised so we'll have a normed_linear layer
            normed_linear = True
            model = MLP(input_size=x_train.shape[1], hidden_size=300, output_size=np.max(y_m_train) + 1, normed_linear=normed_linear, criterion=criterion)
            model.to(device)
            optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
            #equal weight for all samples
            instance_weights = np.ones(y_m_train.shape[0])
            model.fit(x_train, y_m_train, x_dev, y_m_dev, optimizer, instance_weights=instance_weights, n_iter=200, batch_size=1000, max_patience=10)
            f1 = model.score(x_dev, y_m_dev)
            if f1 > best_f1:
                best_f1 = f1
                best_s = s
                best_max_m = max_m
            results[f"{s}"][f"{max_m}"] = f1
    results['ldam_tune'] = results
    print(results)
    return best_f1, best_s, best_max_m

def tune_focal(x_train, y_m_train, x_dev, y_m_dev):
    unique, counts = np.unique(y_m_train, return_counts=True)
    cls_num_list = counts.tolist()
    beta = 0.9999
    effective_num = 1.0 - np.power(beta, cls_num_list)
    per_cls_weights = (1.0 - beta) / np.array(effective_num)
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
    per_cls_weights = variable(torch.FloatTensor(per_cls_weights))
    results = defaultdict(dict)
    best_f1, best_s, best_max_m = 0, 0, 0
    for gamma in [1, 2, 4, 8, 16, 32, 64]:
        criterion = criterion1 = FocalLoss(weight=per_cls_weights, gamma=gamma)
        #only for ldam the last layer weights should be normalised so we'll have a normed_linear layer
        normed_linear = True
        model = MLP(input_size=x_train.shape[1], hidden_size=300, output_size=np.max(y_m_train) + 1, normed_linear=normed_linear, criterion=criterion)
        model.to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
        #equal weight for all samples
        instance_weights = np.ones(y_m_train.shape[0])
        model.fit(x_train, y_m_train, x_dev, y_m_dev, optimizer, instance_weights=instance_weights, n_iter=200, batch_size=1000, max_patience=10)
        f1 = model.score(x_dev, y_m_dev)
        if f1 > best_f1:
            best_f1 = f1
            best_s = gamma
        results[f"{gamma}"] = f1
    results['gamma_tune'] = results
    print(results)
    return best_f1, best_s, best_max_m

def run_all_losses(option='original'):
    results = defaultdict(dict)
    #results[dataset_option][model][measure] = value
    DO_SVM = False
    DO_RANDOM = True
    DO_TUNE_LDAM = False
    DO_TUNE_FOCAL = False
    SAVE_CROSSENTROPY_300D = False
    logging.info('loading train dev test sets...')
    train_data = load_data_deepmoji('../datasets/deepmoji/train', option=option)
    dev_data = load_data_deepmoji('../datasets/deepmoji/dev', option=option)
    test_data = load_data_deepmoji('../datasets/deepmoji/test', option=option)
    x_train, y_p_train, y_m_train = train_data['feature'], train_data['protected_attribute'], train_data['labels']
    x_dev, y_p_dev, y_m_dev = dev_data['feature'], dev_data['protected_attribute'], dev_data['labels']
    x_test, y_p_test, y_m_test = test_data['feature'], test_data['protected_attribute'], test_data['labels']
    logging.info(f'train/dev/test data loaded. X_train: {x_train.shape} X_dev: {x_dev.shape} X_test: {x_test.shape}')

    
    if DO_SVM:
        model = LinearSVC(fit_intercept=True, class_weight='balanced', dual=False, C=0.1, max_iter=10000)
        model.fit(x_train, y_m_train)
        y_test_pred = model.predict(x_test)
        f1 = f1_score(y_m_test, y_test_pred, average='macro')
        #we are not predicting protected attributes from the debiased representations, we are predicting from the original attributes, so the results are not important
        _, biased_diffs = get_TPR(y_m_test, y_test_pred, y_p_test)
        results[option]['svm'] = {"tpr":rms(list(biased_diffs.values()))}
        results[option]['svm'] = {"f1": f1}
        group_results = group_evaluation(y_test_pred, y_m_test, y_p_test)
        results[option]['svm'].update(group_results)

    
    if DO_RANDOM:
        model = DummyClassifier(strategy="most_frequent")
        model.fit(x_train, y_m_train)
        y_test_pred = model.predict(x_test)
        f1 = f1_score(y_m_test, y_test_pred, average='macro')
        _, biased_diffs = get_TPR(y_m_test, y_test_pred, y_p_test)
        results[option]['rand'] = {"tpr": rms(list(biased_diffs.values()))}
        results[option]['rand'].update({"f1": f1})
        group_results = group_evaluation(y_test_pred, y_m_test, y_p_test)
        results[option]['rand'].update(group_results)

    if DO_TUNE_LDAM:
        #tuned and best s was between 0 and 4 e.g. 2
        best_f1, best_s, best_max_m = tune_ldam(x_train, y_m_train, x_dev, y_m_dev)
        print(f"ldam tuned: f1:{best_f1} s:{best_s} max_m:{best_max_m}")
        sys.exit(0)

    if DO_TUNE_FOCAL:
        #tuned and best gamma was 1
        best_f1, best_s, best_max_m = tune_focal(x_train, y_m_train, x_dev, y_m_dev)
        print(f"ldam tuned: f1:{best_f1} s:{best_s} max_m:{best_max_m}")
        sys.exit(0)
    unique, counts = np.unique(y_m_train, return_counts=True)
    cls_num_list = counts.tolist()
    beta = 0.9999
    effective_num = 1.0 - np.power(beta, cls_num_list)
    per_cls_weights = (1.0 - beta) / np.array(effective_num)
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
    per_cls_weights = variable(torch.FloatTensor(per_cls_weights))


    #protected classes effective weight
    unique, counts = np.unique(y_p_train, return_counts=True)
    clsp_num_list = counts.tolist()
    beta = 0.9999
    effective_num = 1.0 - np.power(beta, clsp_num_list)
    per_clsp_weights = (1.0 - beta) / np.array(effective_num)
    per_clsp_weights = per_clsp_weights / np.sum(per_clsp_weights) * len(clsp_num_list)
    per_clsp_weights = variable(torch.FloatTensor(per_clsp_weights))
    #instance_weights = np.ones(y_m_train.shape[0])
    instance_weights = per_clsp_weights[y_m_train].detach().cpu().numpy()

    
    criterion1 = FocalLoss(weight=per_cls_weights, gamma=1)
    criterion2 = LDAMLoss(cls_num_list=cls_num_list, max_m=0.5, weight=per_cls_weights, s=2)
    criterion3 = SelfAdjDiceLoss()
    criterion4 = F.cross_entropy
    criterion5 = CrossEntropyWithInstanceWeights()
    criterion6 = LDAMLossInstanceWeight(cls_num_list=cls_num_list, max_m=0.5, weight=per_cls_weights, s=2)

    criterions = {'ldam':criterion2, 'focal':criterion1, 'adjdice':criterion3, 'crosent':criterion4, 'instanceweights': criterion5, 'ldaminstance': criterion6}
    for c_name, criterion in criterions.items():
        logging.info(f"loss: {c_name}")
        results[option][c_name] = {}
        #only for ldam the last layer weights should be normalised so we'll have a normed_linear layer
        normed_linear = True if c_name == 'ldam' else False
        model = MLP(input_size=x_train.shape[1], hidden_size=300, output_size=np.max(y_m_train) + 1, normed_linear=normed_linear, criterion=criterion)
        model.to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
        #equal weight for all samples
        #instance_weights = np.ones(y_m_train.shape[0])
        model.fit(x_train, y_m_train, x_dev, y_m_dev, optimizer, instance_weights=instance_weights, n_iter=200, batch_size=1000, max_patience=10)
        
        #get the representation from the trained MLP for MLP
        if SAVE_CROSSENTROPY_300D and criterion == criterion4:
            x_train_repr, x_dev_repr, x_test_repr = model.get_hidden(x_train), model.get_hidden(x_dev), model.get_hidden(x_test)
            np.save('../datasets/deepmoji/x_{}_300d.npy'.format('train'), x_train_repr)
            np.save('../datasets/deepmoji/x_{}_300d.npy'.format('dev'), x_dev_repr)
            np.save('../datasets/deepmoji/x_{}_300d.npy'.format('test'), x_test_repr)

        f1 = model.score(x_test, y_m_test)
        y_test_pred = model.predict(x_test)
        _, debiased_diffs = get_TPR(y_m_test, y_test_pred, y_p_test)
        results[option][c_name].update({"f1": f1})
        results[option][c_name].update({"tpr": rms(list(debiased_diffs.values()))})
        group_results = group_evaluation(y_test_pred, y_m_test, y_p_test)
        results[option][c_name].update(group_results)

    return results    


def pretty_print(results, option='original', output_csv_dir='./'):
    for option, res in results.items():
        df = pd.DataFrame(res)
        df.to_csv(os.path.join(output_csv_dir, f"{option}_results.csv"))

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_results = defaultdict(dict)
    for option in ['original']:
        all_results.update(run_all_losses(option=option))
    pretty_print(all_results)
