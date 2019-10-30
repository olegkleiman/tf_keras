import numpy as np
from perceptron import Perceptron
import matplotlib.pyplot as plt

training_inputs = []
# x < y
training_inputs.append(np.array([.5, 1]))
training_inputs.append(np.array([2, 3]))
training_inputs.append(np.array([3.5, 4]))

# x > y
training_inputs.append(np.array([3, 2.1]))
training_inputs.append(np.array([2, 1.1]))
training_inputs.append(np.array([2.2, 1.5]))

labels = np.array([1,1,1,0,0,0])

#
# Plot training set
#
plt.axis([-0.1, 5, -0.1, 5])
plt.gcf().canvas.set_window_title('Neuron classification')   
for sample, label in zip(training_inputs, labels):
    x, y = sample
    color, marker = ('green', '^') if label == 1 else ('red', 's')
    # see about markers: https://matplotlib.org/3.1.1/api/markers_api.html#module-matplotlib.markers
    plt.scatter(x, y, color=color, marker=marker)

#
# Train perceptron
#
perceptron = Perceptron(2)
perceptron.train(training_inputs, labels)
print(perceptron.weights)
bias, w1, w2 = perceptron.weights

#
# print separating line based on training results
#
plt.plot([0, 5], 
          [-bias/w2, (-bias - w1*5)/w2]
        )     
plt.suptitle(f"y={round(-w1/w2, 4)}*x + {round(-bias/w2, 4)}", fontsize='x-large')    

#
# Test
#
input = [.1, 1]
res = perceptron.predict(input)
print(res) # => 1
input = [2, 3.4]
res = perceptron.predict(input)
print(res) # => 1

input = [1.5, .1]
res = perceptron.predict(input)
print(res) # => 0

plt.show()

