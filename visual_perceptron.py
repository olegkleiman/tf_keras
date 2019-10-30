import numpy as np
from sigmoid import sigmoid
from plotable import Plotable

class VisualPerceptron(Plotable):

    def __init__(self, inputs_len, threshold = 100, learning_rate = 0.01):
        Plotable.__init__(self)
        self.weights = np.random.rand(inputs_len + 1)
        # self.weights = np.zeros(inputs_len + 1)
        self.learning_rate = learning_rate
        self.threshold = threshold
        self.finished = False

    def seterror(self, error):
        self.__error = error
    def geterror(self):
        return self.__error
    def delerror(self):
        del self.__error
    error = property(geterror, seterror, delerror)

    #
    # train()
    # External API
    # 
    def visualize_train(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

        super().start(self.__train_step)

    #
    # (private) __train_steps
    # Performs single train iteration over all inputs
    #
    def __train_step(self, i):

        if self.finished:
            return

        is_error = False
        for sample, label in zip(self.inputs, self.labels):
            prediction = self.predict(sample)
            self.error = label - prediction
            is_error = is_error or self.error != 0
            self.weights[1:] += self.learning_rate * self.__error * sample
            self.weights[0] += self.learning_rate * self.error

        super().redraw(self.inputs, self.labels, self.weights, self.error)
        
        if not is_error:
            self.finished = True

    def predict(self, input):
        sum = np.dot(self.weights[1:], input) + self.weights[0]
        return self.__activation(sum)  

    def __activation(self, summation):
        return sigmoid(summation)
        # return 1 if summation > 0 else 0   



