import numpy as np
import math
from sigmoid import sigmoid
from colorama import init, Fore, Back, Style

class Perceptron(object):
    def __init__(self, inputs_len, threshold = 100, learning_rate = 0.01):
        self.weights = np.random.rand(inputs_len + 1) # +1 is for bias
        # self.weights = np.zeros(inputs_len + 1)
        self.learning_rate = learning_rate
        self.threshold = threshold

    def train(self, inputs, labels):
        for i in range(self.threshold):
            j = 1
            for input, label in zip(inputs, labels):
                print("======= Iteration" + Fore.YELLOW + f" {j}", end = ' ')
                print(" Epoch: " + Fore.YELLOW + f"{i}")
                j += 1
                prediction = self.predict(input)
                # error = math.pow(label - prediction, 2)
                error = label - prediction
                print(f"Input: {input} Label: {label} Prediction: {prediction}", end = ' ')
                fore_color = Fore.GREEN if error == 0 else Fore.RED
                print(fore_color + f"Error: {error}")
                print(f"∆w: {self.learning_rate * error} {self.learning_rate * error * input}")
                self.weights[1:] += self.learning_rate * error * input
                self.weights[0] += self.learning_rate * error
                print(f"weights: {self.weights}")

    def predict(self, input):
        sum = np.dot(self.weights[1:], input) + self.weights[0]
        return self.activation(sum)

    def activation(self, summation):
        # return sigmoid(summation)
        return 1 if summation > 0 else 0            

