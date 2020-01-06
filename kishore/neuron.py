import numpy as np
import math

# activation functions
def sigmoid(x):  #  1 / (1 + e^(-x))
    return  1 / (1 + np.exp(-x))

def identity(x):
    return x 

def relu(x): # rectified linear unit
    return np.where(x>0, x, 0)

def tanh(x): 
    return (math.exp(x)-math.exp(-x))/(math.exp(x)+math.exp(-x)) 

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x)) 

# loss functions
def squared_error(x, y):
    return np.square(x - y)

def mse(p, y): 
    return np.mean(np.square(p-y)) 

def mae(p, y): 
    return np.mean(np.abs(p-y))

def cat_cross_entropy(p, y): 
    return np.sum((y*np.log2(p)+(1-y)*np.log2(1-p)))

class Model:
    def __init__(self):
        self.hidden = []
        self.output = {}

    def add(self, neuron):
        self.hidden.append(neuron)

    def add_output(self, neuron):
        self.output = neuron

    def train(self, inputs, labels):
        for (sample, label) in zip(inputs, labels):
            layer_output = []
            for x in self.hidden:
                layer_output.append(x.predict(sample))

            res = self.output.predict(layer_output)
            print(res)
            loss = squared_error(res, label)
            print(loss)

class Neuron:
    def __init__(self, weights, activation=sigmoid):
        
        self.weights = []
        self.weights = weights
        if isinstance(activation, str):
            switcher = {
                'sigmoid': sigmoid,
                'identity': identity,
                'relu': relu,
                'tanh': tanh,
                'softmax': softmax
            }
            self.activation = switcher.get(activation)
        else:
           self.activation = activation     

    def predict(self, inputs):
        product = np.dot(self.weights, inputs)
        ret = round(self.activation(product), 2)
        print(ret)
        return ret

inputs = []
inputs.append([1.,1.])
labels = [0]

model = Model()

model.add(Neuron([.8,.2]))
model.add(Neuron([.4,.9]))
model.add(Neuron([.3,.5]))

model.add_output(Neuron([.3,.5,.9], activation='identity'))

model.train(inputs, labels)
