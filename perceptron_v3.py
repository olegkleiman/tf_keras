import tensorflow as tf
import numpy as np

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

x = .8 # np.array([0.8])
y = 0. # np.array([0.])

class Model(object):
    def __init__(self, x, y):
        self.W = tf.Variable(x, name="weigth")
        self.bias = tf.Variable(0.)

    def __call__(self, x):
        return self.W * x # + self.bias

def loss(predicted_y, desired_y):
    return tf.reduce_sum(tf.square(predicted_y - desired_y)) # reduce across all axes
    # return tf.square(predicted_y - desired_y)

# optimizer = tf.optimizers.Adam(0.1)
optimizer = tf.optimizers.SGD(learning_rate = 0.025)

def train(model, inputs, outputs):
    with tf.GradientTape() as tape:
        current_loss = loss(model(inputs), outputs)
    grads = tape.gradient(current_loss, [model.W, model.bias])
    optimizer.apply_gradients(zip(grads,[model.W, model.bias]))
    print(f"Current loss: {current_loss}, weigth: {model.W.value()}")

model = Model(x, y)   

for i in range(300):
    print(model.bias.numpy())
    train(model,x,y)
