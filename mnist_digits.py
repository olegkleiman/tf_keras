import datetime
from dataclasses import dataclass

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import random

@dataclass
class CNN_MNIST:

    def __init__(self):
        mnist = tf.keras.datasets.mnist 
        (x_train, y_train), (x_test, y_test) = mnist.load_data(path='mnist.npz')

        self.xTrain, self.yTrain = x_train, y_train
        self.xTest, self.yTest = x_test, y_test

        log_dir = "./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tb_callback = keras.callbacks.TensorBoard(log_dir=log_dir)

    def createModel(self): 
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

    def prepare(self, epochs = 5):
        self.model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
        self.model.fit(self.xTrain, 
                        self.yTrain, 
                        epochs=epochs,
                        validation_data=(self.xTest, self.yTest),
                        callbacks=[self.tb_callback])
        self.model.evaluate(self.xTest, self.yTest , verbose=2)      

    def predict(self, imageIndex): 
        img_rows = 28
        img_cols = 28
        pred = self.model.predict(self.xTest [imageIndex].reshape(1, img_rows, img_cols))  
        self.pred = pred.argmax()
        return self.pred
    
    def __str__(self):
        return f"{self.pred}"

cnn = CNN_MNIST()
cnn.createModel()
cnn.prepare(epochs = 5)
image_index = random.randint(0, 9999)
print("Image index: " + str(image_index))
prediction = cnn.predict(image_index)
print('Prediction: ' + str(cnn))
print('Test: ', str(cnn.yTest[image_index]))
plt.imshow(cnn.xTest[image_index], cmap='Greys')
plt.show()