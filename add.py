import numpy as np
from perceptron import Perceptron 

training_inputs = []
training_inputs.append(np.array([1, 1]))
training_inputs.append(np.array([1, 0]))
training_inputs.append(np.array([0, 1]))
training_inputs.append(np.array([0, 0]))

# AND labels
labels = np.array([1, 0, 0, 0])

perceptron = Perceptron(2)
perceptron.train(training_inputs, labels)

inputs = np.array([1, 1])
res = perceptron.predict(inputs) 
print(f"1^1={res}") #=> 1

inputs = np.array([0, 1])
res = perceptron.predict(inputs) 
print(f"0^1={res}") #=> 0

inputs = np.array([1, 0])
res = perceptron.predict(inputs) 
print(f"1^0={res}") #=> 0

inputs = np.array([0, 0])
res = perceptron.predict(inputs) 
print(f"1^0={res}") #=> 0 