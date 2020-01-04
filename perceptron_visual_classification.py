import numpy as np
from visual_perceptron import VisualPerceptron
import sys, os

from random import seed
from random import random

training_inputs = []
labels = []

for i in range(8):
    x = round(random() * 10, 2)
    y = round(random() * 10, 2)
    label = 1 if x < y else 0
    labels.append(label)
    training_inputs.append([x,y])

# input_filename = os.path.dirname(os.path.abspath(sys.argv[0])) + "/train_data.txt"
# training_inputs = np.loadtxt(input_filename, comments='#', delimiter=',', dtype=np.float32)

# labels_filename = os.path.dirname(os.path.abspath(sys.argv[0])) + "/labels.txt"
# labels = np.loadtxt(labels_filename, comments='#', delimiter=',', dtype=np.float32)

p = VisualPerceptron(2, activation='sigmoid') 
p.visualize_train(training_inputs, labels)