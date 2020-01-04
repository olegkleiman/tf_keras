import numpy as np
from sigmoid import sigmoid
from plotable import Plotable

# activation functions
def _sigmoid(x):  #  1 / (1 + e^(-x))
    return  1 / (1 + np.exp(-x))

def _swish(x, beta = 1.):
    return x * _sigmoid(beta * x)

def _identity(x):
    return x 

def _relu(x): # rectified linear unit
    return np.where(x>0, x, .0)

def _tanh(x): 
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x)) 

def _softmax(x):
    return np.exp(x)/np.sum(np.exp(x)) 

# loss functions
def _squared_error(x, y):
    return np.square(x - y)

def _mse(p, y): 
    return np.mean(np.square(p-y)) 

def _mae(p, y): 
    return np.mean(np.abs(p-y))

def _cat_cross_entropy(p, y): 
    return np.sum((y*np.log2(p)+(1-y)*np.log2(1-p)))

class VisualPerceptron(Plotable):

    def __init__(self, inputs_len, activation='sigmoid',
                 threshold = 100, learning_rate = 0.01):
        Plotable.__init__(self)

        if isinstance(activation, str):
            switcher = {
                'sigmoid': _sigmoid,
                'identity': _identity,
                'relu': _relu,
                'tanh': _tanh,
                'softmax': _softmax
            }
            self.activation = switcher.get(activation)
        else:
           self.activation = activation   

        # self.weights = np.random.rand(inputs_len + 1)
        self.weights = np.array([0.1,0.1,0.1])
        # self.weights = np.zeros(inputs_len + 1)
        self.learning_rate = learning_rate
        self.threshold = threshold
        self.finished = False
        self.loss = .0

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
        step = 1
        for sample, label in zip(self.inputs, self.labels):
            prediction = self.predict(sample)
            self.error = label - prediction
            self.loss = self.loss + np.square(self.error)
            is_error = is_error or self.error != 0

            X = np.insert(sample, 0, [1]) # adding bias
            self.weights += np.multiply(self.learning_rate * self.__error, X)

            # self.weights[1:] += np.multiply(self.learning_rate * self.__error, sample)
            # self.weights[0] += np.multiply(self.learning_rate, self.error)
            step = step + 1

        super().redraw(self.inputs, self.labels, self.weights, self.error, self.loss)
        self.loss = round(self.loss / step, 4)
        if self.loss < 0.01:
            self.finished = True
        
        
        if not is_error:
            self.finished = True

    def predict(self, input):
        sum = np.dot(self.weights[1:], input) + self.weights[0]
        return self.activation(sum)  

    def __activation(self, summation):
        return sigmoid(summation)
        # return 1 if summation > 0 else 0   



