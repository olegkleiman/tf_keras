import matplotlib.pyplot as plt
import numpy as np

class operation(object):

    def __init__(self, threshold=100, learning_rate=0.01):
        self.threshold = threshold
        self.weights = np.zeros(3) # bias is 0-th weight
        self.learning_rate = learning_rate

    def activation(self, res):
        return 1 if res > 0 else 0

    def __call__(self, inputs):
        _res= np.dot(inputs, self.weights[1:]) + self.weights[0]
        return  self.activation(_res)


training_inputs = []
training_inputs.append(np.array([1, 1]))
training_inputs.append(np.array([1, 0]))
training_inputs.append(np.array([0, 1]))
training_inputs.append(np.array([0, 0]))

labels = np.array([1, 0, 0, 0])

add = operation()
# for _ in range(add.threshold):
for inputs, label in zip(training_inputs, labels):
    prediction = add(inputs)
    print(f"w   = {add.weights[1:]} bias: {add.weights[0]}")
    add.weights[1:] += add.learning_rate * (label - prediction) *inputs
    add.weights[0] += add.learning_rate * (label - prediction)
    print(f"Input: {inputs} Prediction: {prediction} Label: {label}")
    print(f"Updated W: {add.weights[1:]} bias: {add.weights[0]}")

# predict = add(training_inputs[0])
