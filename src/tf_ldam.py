import tensorflow as tf
import numpy as np
import pdb




class NormedLinear(keras.layers.Layer):
    #for ldam, not exactly similar to pytorch code as in pytorch code maxnorm is applied on the initialized weights.
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()
        w_init = tf.random_uniform_initializer(minval=-1, maxval=1)
        self.w = tf.Variable(
            initial_value=w_init(shape=(in_features, out_features), dtype="float32"),
            trainable=True,
        )

    def call(self, inputs):
        return tf.matmul(tf.keras.utils.normalize(inputs, axis=-1, order=2), tf.keras.utils.normalize(self.w, axis=-1, order=2))





def ldamloss(x=None, targets=None, cls_num_list=[100, 1], max_m=0.5, weight=None, s=30):
    #just sample values
    weights = tf.convert_to_tensor(cls_num_list, dtype=tf.float32)
    x = tf.random.uniform((10, 2), minval=0, maxval=10, dtype=tf.dtypes.float32, seed=0, name=None)
    targets = tf.convert_to_tensor([0, 1, 0, 1, 1, 1, 1, 1, 1, 1], dtype=tf.dtypes.int32)
    
    
    num_class = len(cls_num_list)
    m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
    m_list = m_list * (max_m / np.max(m_list))
    m_list = tf.convert_to_tensor(m_list, dtype=tf.float32)

    
    

    index_int = tf.one_hot(targets, depth=num_class, on_value=1, off_value=0, dtype=tf.dtypes.int32)
    index_float = tf.cast(index_int, dtype=tf.dtypes.float32)
    index_bool = tf.cast(index_int, dtype=tf.dtypes.bool)
    m_list = tf.reshape(m_list, (num_class, 1))
    batch_m = tf.matmul(index_float, m_list)
    x_m = x - batch_m
    output = tf.where(index_bool, x_m, x)
    
    if weight is not None:
        sample_weights = tf.gather(weights, targets)
    else:
        sample_weights = None
    
    cce = tf.keras.losses.CategoricalCrossentropy()
    pdb.set_trace()
    return cce(index, output * s, sample_weight=sample_weights)


ldamloss()