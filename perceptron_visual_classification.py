import numpy as np
from visual_perceptron import VisualPerceptron
import sys, os

input_filename = os.path.dirname(os.path.abspath(sys.argv[0])) + "/train_data.txt"
training_inputs = np.loadtxt(input_filename, comments='#', delimiter=',', dtype=np.float32)

labels_filename = os.path.dirname(os.path.abspath(sys.argv[0])) + "/labels.txt"
labels = np.loadtxt(labels_filename, comments='#', delimiter=',', dtype=np.float32)

p = VisualPerceptron(2) 
p.visualize_train(training_inputs, labels)