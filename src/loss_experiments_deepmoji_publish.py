import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans, AgglomerativeClustering
from collections import Counter, defaultdict
import json
from collections import defaultdict
import sys
import os
import torch
from losses import *
from models import MLP, variable, MLP_adv
import pickle
import logging
import random
import pandas as pd
import pdb
import sys
sys.path.append('../')

from data.process_data import load_data_deepmoji, upsample, load_data_biasbios

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

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

def run_all_losses(option='original', class_balance=0.5):
    results = defaultdict(dict)
    DO_RANDOM = True
    SAVE_CROSSENTROPY_300D = False
    
    logging.info('loading train dev test sets...')
    train_data = load_data_deepmoji('/home/sssub/classimb_fairness-hiddenreps/datasets/deepmoji/train', option=option, class_balance=class_balance)
    dev_data = load_data_deepmoji('/home/sssub/classimb_fairness-hiddenreps/datasets/deepmoji/dev', option=option, class_balance=class_balance)
    test_data = load_data_deepmoji('/home/sssub/classimb_fairness-hiddenreps/datasets/deepmoji/test', option=option, class_balance=class_balance)

    x_train, y_p_train, y_m_train = train_data['feature'], train_data['protected_attribute'], train_data['labels']
    x_dev, y_p_dev, y_m_dev = dev_data['feature'], dev_data['protected_attribute'], dev_data['labels']
    x_test, y_p_test, y_m_test = test_data['feature'], test_data['protected_attribute'], test_data['labels']

    logging.info(f'train/dev/test data loaded. X_train: {x_train.shape} X_dev: {x_dev.shape} X_test: {x_test.shape}')

    if DO_RANDOM:
        model = DummyClassifier(strategy="stratified")
        model.fit(x_train, y_m_train)
        y_test_pred = model.predict(x_test)
        f1 = f1_score(y_m_test, y_test_pred, average='macro')
        _, biased_diffs = get_TPR(y_m_test, y_test_pred, y_p_test)
        results[option]['rand'] = {"tpr": rms(list(biased_diffs.values()))}
        results[option]['rand'].update({"f1": f1})
        group_results = group_evaluation(y_test_pred, y_m_test, y_p_test)
        results[option]['rand'].update(group_results)

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

    criterion1 = FocalLoss(weight=per_cls_weights, gamma=1)
    criterion2 = SelfAdjDiceLoss()
    criterion3 = F.cross_entropy 
    criterion4 = CrossEntropyWithInstanceWeights() #iw
    criterion5 = CrossEntropyWithInstanceWeights(class_weights=per_cls_weights) #cw
    criterions = {
        'focal':criterion1, 
        'adjdice':criterion2, 
        'CE': criterion3,
        'iw':criterion4, 
        'cw':criterion5
    }
    
    for _s in [0.5, 1, 3, 5, 10, 15, 20, 25, 30]:
        ldamcriterion = LDAMLoss(cls_num_list=cls_num_list, max_m=0.5, weight=None, s=_s) #ldam
        ldamcwcriterion = LDAMLoss(cls_num_list=cls_num_list, max_m=0.5, weight=per_cls_weights, s=_s) #ldam cw
        
        ldamiw = GeneralLDAMLoss(cls_num_list=cls_num_list, clsp_num_list=clsp_num_list, 
                mp_num_list=pm_counts, max_m=0.5, class_weight=None, ldams=_s, ldamc=1, 
                ldamg=0, ldamcg=0, use_instance=True, ldam_mul_c_g=False) #ldamiw

        criterions['criteria_ldam_{}'.format(_s)] = ldamcriterion
        criterions['criteria_ldam_cw_{}'.format(_s)] = ldamcwcriterion
        criterions['criteria_ldam_ldamiw_{}'.format(_s)] = ldamiw
        
        for _rho in [0.01, 0.1, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 50, 70, 90, 100]:
            ldamreg = GeneralLDAMLoss(cls_num_list=cls_num_list, clsp_num_list=clsp_num_list, 
                    mp_num_list=pm_counts, max_m=0.5, class_weight=None, ldams=_s, ldamc=1, 
                    ldamg=0, ldamcg=0, use_instance=False, ldam_mul_c_g=False, rho=_rho)
            
            criterions['criteria_ldam_ldamreg_{}_{}'.format(_s, _rho)] = ldamreg
        
    """for c_name, criterion in criterions.items():
        set_seed(SEED)
        logging.info(f"loss: {c_name}")
        results[option][c_name] = {}
        normed_linear = True if 'ldam' in c_name.lower() else False
        model = MLP(input_size=x_train.shape[1], hidden_size=300, output_size=np.max(y_m_train) + 1, normed_linear=normed_linear, criterion=criterion)
        model.to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
        model.fit(x_train, y_m_train, y_p_train, y_mp_train, x_dev, y_m_dev, optimizer, instance_weights=instance_weights, n_iter=200, batch_size=1000, max_patience=10)
        f1 = model.score(x_test, y_m_test)
        y_test_pred = model.predict(x_test)
        _, debiased_diffs = get_TPR(y_m_test, y_test_pred, y_p_test)
        results[option][c_name].update({"f1": f1})
        results[option][c_name].update({"tpr": rms(list(debiased_diffs.values()))})
        group_results = group_evaluation(y_test_pred, y_m_test, y_p_test)
        results[option][c_name].update(group_results)"""

    for _s in [0.5, 1, 3, 5, 10, 15, 20, 25, 30]:
        for adv_val in [0.01, 0.1, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 50, 70, 90, 100]:
            logging.info(f"adv: {_s}_{adv_val}")
            normed_linear = True
            set_seed(SEED)
            adv_ldamcriterion = LDAMLoss(cls_num_list=cls_num_list, max_m=0.5, weight=None, s=_s) #ldam
            #advmodel = MLP_adv(input_size=x_train.shape[1], hidden_size=300, output_size=np.max(y_m_train) + 1, domain_output_size = np.max(y_p_train) + 1, normed_linear=normed_linear, criterion1=adv_ldamcriterion, criterion2=F.cross_entropy, alpha=adv_val)
            advmodel = MLP_adv(input_size=x_train.shape[1], hidden_size=300, output_size=np.max(y_m_train) + 1, domain_output_size = np.max(y_p_train) + 1, normed_linear=normed_linear, criterion1=adv_ldamcriterion, criterion2=F.cross_entropy, alpha=adv_val, rev_tech=0)
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
            group_results = group_evaluation(y_test_pred, y_m_test, y_p_test)
            results[option]["adv_{}_{}".format(_s, adv_val)].update(group_results)
            
            logging.info(f"f1: {f1}")
            logging.info(f"tpr: {_tpr_val}")
    
    return results    

def pretty_print(results, option='original', output_csv_dir='./', class_balance=0.5):
    for option, res in results.items():
        df = pd.DataFrame(res)
        df.to_csv(os.path.join(output_csv_dir, f"{option}_{class_balance}_results_apr15_adv_newgradrev_alpha_seed{SEED}.csv"))

if __name__ == "__main__":

    #datasets_to_run = ['deepmoji', 'biography']
    datasets_to_run = ['deepmoji']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"using device {device}")

    for ds in datasets_to_run:
        if ds == 'deepmoji':
            all_results = defaultdict(dict)
            all_results.update(run_all_losses(option='original'))
            pretty_print(all_results)
            cb = [0.9, 0.95]
            option = ['inlp0.9', 'inlp0.95']
            for _i in range(len(cb)):
                all_results = defaultdict(dict)
                all_results.update(run_all_losses(option=option[_i], class_balance=cb[_i]))
                pretty_print(all_results, class_balance=cb[_i])
        elif ds == "biography":
            all_results = defaultdict(dict)
            all_results.update(run_all_losses_biasbios())
            pretty_print_biography(all_results)
  
            
            
        