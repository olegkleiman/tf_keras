import numpy as np
from colorama import init, Fore, Back, Style
from perceptron import Perceptron
import matplotlib.pyplot as plt

plt.switch_backend('macosx')
init(autoreset=True)

# fig = plt.figure()

training_inputs = []
training_inputs.append(np.array([1, 1]))
training_inputs.append(np.array([1, 0]))
training_inputs.append(np.array([0, 1]))
training_inputs.append(np.array([0, 0]))

# OR labels
labels = np.array([1, 1, 1, 0])

#
# Plot training set
#
for sample, label in zip(training_inputs, labels):
    x, y = sample
    color, marker = ('green', '^') if label == 1 else ('red', 's')
    # see about markers: https://matplotlib.org/3.1.1/api/markers_api.html#module-matplotlib.markers
    plt.scatter(x, y, color=color, marker=marker)

# plt.axis([-0.1, 1.1, -0.1, 1.1])

perceptron = Perceptron(2, threshold = 100)
perceptron.train(training_inputs, labels)
print(perceptron.weights)
bias = perceptron.weights[0]
plt.plot([0, 1], 
          [bias, perceptron.weights[1] + bias]
        )

input = [1,1]
res = perceptron.predict(input)
print(Style.BRIGHT + Fore.BLUE + f"1|1={res}")

input = [1,0]
res = perceptron.predict(input)
print(Style.BRIGHT + Fore.BLUE + f"1|0={res}") # => 1

input = [0,1]
res = perceptron.predict(input)
print(Style.BRIGHT + Fore.BLUE + f"0|1={res}") # => 1

input = [0,0]
res = perceptron.predict(input)
print(Style.BRIGHT + Fore.BLUE + f"0|0={res}") # => 0

plt.show()