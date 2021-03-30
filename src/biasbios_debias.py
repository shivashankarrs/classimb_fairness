"""
Usage:
  deepmoji_debias.py [--input_dir=INPUT_DIR] [--output_dir=OUTPUT_DIR] [--in_dim=IN_DIM] [--n=N]
Options:
  -h --help                     show this help message and exit
  --input_dir=INPUT_DIR         input dir file
  --output_dir=OUTPUT_DIR       write down output file
  --in_dim=IN_DIM               input dimension of the vectors [default: 300]
  --n=N                         number of epochs to run classifier
Script for learning a debiasing matrix P from some vectors data
"""

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.utils import shuffle
from sklearn.svm import LinearSVC
import debias


def find_projection_matrices(X_train, Y_train_protected, X_dev, Y_dev_protected, Y_train_main, Y_dev_main,
                             dim, out_dir, n, name):
    is_autoregressive = True
    min_acc = 0.51
    noise = False

    print("num classifiers: {}".format(n))

    #clf = SGDClassifier
    #params = {'warm_start': True, 'loss': 'log', 'n_jobs': -1, 'max_iter': 1200, 'random_state': 0, 'tol': 1e-3}
    clf = LinearSVC
    params = {'fit_intercept': True, 'class_weight': 'balanced', 'dual': False, 'C': 0.1}

    P_n = debias.get_debiasing_projection(clf, params, n, dim, is_autoregressive, min_acc,
                                          X_train, Y_train_protected, X_dev, Y_dev_protected,
                                          by_class=True, Y_train_main=Y_train_main, Y_dev_main=Y_dev_main)
    fname = out_dir + "/P_svm.num-clfs={}_{}.npy".format(n, name)
    np.save(fname, P_n)


if __name__ == '__main__':
    in_dir = '/home/sssub/classimb_fairness-hiddenreps/datasets/bios/'
    out_dir = '/home/sssub/classimb_fairness-hiddenreps/datasets/bios/projection/'
    in_dim = 300
    n = 300

    import pickle
    with open(in_dir+'bios_train_with_hidden.pickle', 'rb') as f:
        train_data = pickle.load(f)

    with open(in_dir+'bios_dev_with_hidden.pickle', 'rb') as f:
        dev_data = pickle.load(f)

    with open(in_dir+'bios_test_with_hidden.pickle', 'rb') as f:
        test_data = pickle.load(f)
        
    x_train, y_train_protected, y_train_main = train_data['hidden'], train_data['protected_attribute'], train_data['labels']
    x_dev, y_dev_protected, y_dev_main = dev_data['hidden'], dev_data['protected_attribute'], dev_data['labels']
    x_test, y_test_protected, y_test_main = test_data['hidden'], test_data['protected_attribute'], test_data['labels']

    find_projection_matrices(x_train, y_train_protected, x_dev, y_dev_protected, y_train_main, y_dev_main, in_dim, out_dir, n, "bios")
