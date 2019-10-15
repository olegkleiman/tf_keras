# tf_keras
TensorFlow2 with Keras
CNNs with different models built with Keras on TensorFlow2

## Content
1. TF2 eager execution. Constants and variables. @tf.function decorator
2. [Simple percepton](https://github.com/olegkleiman/tf_keras/blob/master/perceptron_v1.py)(TF1): loss function, optimizer
3. [Yet Another Simple Perceptron](https://github.com/olegkleiman/tf_keras/blob/master/perceptron_v2.py)(TF1): optimizer with minimize()
4. Multi-dimentional perceptron
5. Keras: layers of perceptrons
6. Models: Keras.Sequential()

## How to build
1. Use Python virtual environment (.venv) 
2. TensorFlow callbacks are served by [TensorBoard](https://www.tensorflow.org/tensorboard/get_started): $tensorboard --logdir logs/fit. Observe it at http://localhost:6000

## Notes
MNIST datasets are used in this exercises: MNIST digist - a set of handwritten 60.000 digits (28x28 px) for train set and 10.000 digits for test set, MNIST Fashion - images of clothes (same size)