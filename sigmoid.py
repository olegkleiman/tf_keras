import tensorflow as tf
import numpy as np

def sigmoid(x):  #  1 / (1 + e^(-x))
    return tf.truediv(tf.constant(1.0), 
                 tf.add(tf.constant(1.0), np.float32(tf.exp(tf.negative(x))))
    )
