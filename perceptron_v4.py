import datetime
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# from keras import backend as K
import sys, os
from sigmoid import sigmoid

input_filename = os.path.dirname(os.path.abspath(sys.argv[0])) + "/train_data.txt"
training_inputs = np.loadtxt(input_filename, comments='#', delimiter=',', dtype=np.float32)

labels_filename = os.path.dirname(os.path.abspath(sys.argv[0])) + "/labels.txt"
labels = np.loadtxt(labels_filename, comments='#', delimiter=',', dtype=np.float32)

log_dir = "./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tb_callback = keras.callbacks.TensorBoard(log_dir=log_dir)

class XModel(tf.keras.Model):
    def __init__(self, layer):
        super(XModel, self).__init__()
        self.classifier = layer

    def call(self, inputs):
        return self.classifier(inputs)

class Linear(layers.Layer):
    def __init__(self, output_dim=1, input_dim=2):
        self.output_dim = output_dim
        super(Linear, self).__init__()
        # self.units = units
        # w_init = tf.random_normal_initializer()
        # self.w = tf.Variable(initial_value=w_init(shape=(input_dim, units),
        #                                          dtype='float32'),
        #                     trainable=True)
        # b_init = tf.zeros_initializer()
        # self.b = tf.Variable(initial_value=b_init(shape=(units,),
        #                                       dtype='float32'),
        #                  trainable=True)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(Linear, self).build(input_shape)
    
    def call(self, inputs):
    # def __call__(self, inputs):
        # print(self.kernel.value())
        
        return keras.backend.dot(inputs, self.kernel)
        # return self.__activation(tf.tensordot(inputs, self.w, 1) + self.b)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    # @tf.function
    # def __activation(self, summation):
    #     # return sigmoid(summation)
    #     return 1 if summation > 0 else 0   


# model = XModel([
#     Linear(1, 2)
# ])

layer = Linear(2, 1)
model = tf.keras.Sequential([
    # tf.keras.layers.Dense(units=2, activation='sigmoid', name='input_dense')
    layer
])

model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
tf.keras.utils.plot_model(model, './perceptron_v4.png')

# Train
model.fit(training_inputs, 
        labels, 
        epochs=5,
        callbacks=[tb_callback]) 

# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()
print(layer.get_weights())

pred = model.predict([[6.,7.]])
# res = pred.argmax()
print(pred)