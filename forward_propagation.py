import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sigmoid import sigmoid

x = tf.constant(.8) # np.array([0.8])
y = tf.Variable(1.6) # np.array([0.])
w = tf.Variable(0.)
b = tf.Variable(0.)

z_1 = tf.add(x*w, b) # tf.add(tf.matmul(x, w), b)
print(f"value: {z_1}")
a_1 = sigmoid(z_1)
print(f"sigmoid: {a_1}")

for x in np.arange(-4., 4., .2):
    _sigmoid = sigmoid(x)
    print(f"x: {x} sigma: {_sigmoid}")
    plt.plot(x, _sigmoid, 'ro')

plt.show()