import os
import pickle
import numpy as np
from sklearn.metrics import f1_score
from tensorflow.keras.layers import Input, Embedding
from tensorflow.keras.layers import Bidirectional, GRU
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
import tensorflow as tf
from math import exp
import keras.backend as K
from tensorflow.keras.layers import Layer
from tensorflow.python.framework import ops
import pandas as pd 
from tensorflow.keras.callbacks import ModelCheckpoint
import sys
sys.path.append('../')
from tensorflow.keras.models import load_model
from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()
from data.process_data import load_data_deepmoji, load_data_biasbios

tf.random.set_seed(1)
os.environ['PYTHONHASHSEED']=str(1)
np.random.seed(1)

LAMBDA_REVERSAL_STRENGTH = 1

import pdb
def ldamloss(cls_num_list=[100, 1], max_m=0.5, weight=None, s=30):

    def lossFunction(y_true, y_pred):    
        num_class = len(cls_num_list)
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = tf.convert_to_tensor(m_list, dtype=tf.float32)
        index = tf.one_hot(y_true, depth=num_class, on_value=1, off_value=0, dtype=tf.dtypes.int32)
        index_float = tf.cast(index, dtype=tf.dtypes.float32)
        index_bool = tf.cast(index, dtype=tf.dtypes.bool)
        m_list = tf.reshape(m_list, (num_class, 1))
        batch_m = tf.matmul(index_float, m_list)
        x_m = x - batch_m
        output = tf.where(index_bool, x_m, x)
        if weight is not None:
            sample_weights = tf.gather(weights, y_true)
        else:
            sample_weights = None
        cce = tf.keras.losses.CategoricalCrossentropy()
        pdb.set_trace()
        return cce(index, output * s, sample_weight=sample_weights)

    return lossFunction
    
def regularized_model(protected_labels):
    def lossFunction(y_true, y_pred):
        y_true = tf.cast(y_true, 'float32')
        #loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        class_1_pred = y_true * y_pred
        class_0_pred = (1 - y_true) * y_pred
        diff = (tf.reduce_mean(class_1_pred) - tf.matmul(class_1_pred, protected_labels)) + (tf.reduce_mean(class_1_pred) - tf.matmul(class_1_pred, 1-protected_labels))
        diff += (tf.reduce_mean(class_0_pred) - tf.matmul(class_0_pred, protected_labels)) + (tf.reduce_mean(class_0_pred) - tf.matmul(class_0_pred, 1-protected_labels))
        loss = tf.reduce_mean(tf.square(diff))
        return loss
    return lossFunction


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

def reverse_gradient(X, hp_lambda):
    """Flips the sign of the incoming gradient during training."""
    try:
        reverse_gradient.num_calls += 1
    except AttributeError:
        reverse_gradient.num_calls = 1

    grad_name = "GradientReversal%d" % reverse_gradient.num_calls

    @ops.RegisterGradient(grad_name)
    def _flip_gradients(grad):
        return [tf.negative(grad) * hp_lambda]
    #g = K.get_session().graph
    g = tf.compat.v1.Session().graph
    with g.gradient_override_map({'Identity': grad_name}):
        y = tf.identity(X)

    return y

class GradientReversal(Layer):
    """Layer that flips the sign of gradient during training."""

    def __init__(self, hp_lambda, **kwargs):
        super(GradientReversal, self).__init__(**kwargs)
        self.supports_masking = True
        self.hp_lambda = hp_lambda

    @staticmethod
    def get_output_shape_for(input_shape):
        return input_shape

    def build(self, input_shape):
        self._trainable_weights = []

    def call(self, x, mask=None):
        return reverse_gradient(x, self.hp_lambda)

    def get_config(self):
        config = {}
        base_config = super(GradientReversal, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def build_dnn(trainX, trainy, train_priv, validX, validy, valid_priv, hidden_size=300, input_dim=2304, epochs=10, patience=3):

    unique, counts = np.unique(trainy, return_counts=True)
    cls_num_list = counts.tolist()
    
    text_input = Input(
        shape=(input_dim,), dtype='float32', name='input'
    )
    embeds = Dense(
        hidden_size, activation='relu'
    )(text_input) 

    predicts = Dense(
        1, activation='sigmoid', name='predict'
    )(embeds) 

    gend_flip_layer = GradientReversal(LAMBDA_REVERSAL_STRENGTH)
    gend_in = gend_flip_layer(embeds)
    gend_out = Dense(
        units=1, activation='sigmoid', name='demo_classifier'
        )(gend_in)

    model = Model(inputs=text_input, outputs=[predicts, gend_out])#eth_out
    
    layer_names = ['predict', 'demo_classifier']
    loss_dict = {}
    metrics_dict = {}

    for l in layer_names: 
        if l == 'predict':
            loss_dict[l] = ldamloss(cls_num_list=cls_num_list)
        else: 
            loss_dict[l] = 'binary_crossentropy'
        metrics_dict[l] = 'accuracy'
    
    model.compile(
        loss=loss_dict, optimizer='rmsprop',
        metrics=metrics_dict, loss_weights = [1, 1]
    )
    print(model.summary())
    #mc = ModelCheckpoint('adv_best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
    #history = model.fit(trainX, [trainy, train_priv], validation_data=(validX, [validy, valid_priv]), epochs=epochs, verbose=0, callbacks=[mc], batch_size=32)
    history = model.fit(trainX, [trainy, train_priv], validation_data=(validX, [validy, valid_priv]), epochs=epochs, verbose=0, batch_size=32)
    return model
    

def build_reg_dnn(trainX, trainy, train_priv, validX, validy, valid_priv, hidden_size=300, input_dim=2304, epochs=10, patience=3):

    text_input = Input(
        shape=(input_dim,), dtype='float32', name='input'
    )
    protected_labels = Input(
        shape=(1,), dtype='float32', name='prot_input'
    )

    embeds = Dense(
        hidden_size, activation='relu'
    )(text_input) 
    predicts = Dense(
        1, activation='sigmoid', name='predict'
    )(embeds) 

    model = Model(inputs=[text_input, protected_labels], outputs=predicts)#eth_out
    
    model.compile(
        loss=regularized_model(protected_labels), optimizer='rmsprop'
    )
    print(model.summary())
    mc = ModelCheckpoint('reg_best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
    history = model.fit([trainX, train_priv], trainy, validation_data=([validX, valid_priv], validy), epochs=epochs, verbose=0, callbacks=[mc], batch_size=32)
    return model
    
if __name__ == '__main__':
    class_balance = 0.9
    option = 'inlp0.5'
    train_data = load_data_deepmoji('../datasets/deepmoji/train', option=option, class_balance=class_balance)
    dev_data = load_data_deepmoji('../datasets/deepmoji/dev', option=option, class_balance=class_balance)
    test_data = load_data_deepmoji('../datasets/deepmoji/test', option=option, class_balance=class_balance)
    x_train, y_p_train, y_m_train = train_data['feature'], train_data['protected_attribute'], train_data['labels']
    x_dev, y_p_dev, y_m_dev = dev_data['feature'], dev_data['protected_attribute'], dev_data['labels']
    x_test, y_p_test, y_m_test = test_data['feature'], test_data['protected_attribute'], test_data['labels']
    advmodel = build_dnn(x_train, y_m_train, y_p_train, x_dev, y_m_dev, y_p_dev, hidden_size=300, input_dim=2304, epochs=200, patience=10)
    #saved_model = load_model('adv_best_model.h5')
    y_test_pred = advmodel.predict(x_test)
    f1 = f1_score(y_m_test, y_test_pred, average='macro')
    _, biased_diffs = get_TPR(y_m_test, y_test_pred, y_p_test)
    print ("tpr:", rms(list(biased_diffs.values())), "f1:",  f1)
    
