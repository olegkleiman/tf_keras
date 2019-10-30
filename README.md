# tf_keras
TensorFlow2 with Keras
CNNs with different models built with Keras on TensorFlow2

## Content
1. [Simplest (but visualized) perceptron](https://github.com/olegkleiman/tf_keras/blob/master/perceptron_visual_classification.py) with Python only
2. TF2 eager execution. Constants and variables. @tf.function decorator
3. [Simple percepton](https://github.com/olegkleiman/tf_keras/blob/master/perceptron_v1.py)(TF1): loss function, optimizer
4. [Yet Another Simple Perceptron](https://github.com/olegkleiman/tf_keras/blob/master/perceptron_v2.py)(TF1): optimizer with minimize()
5. [Perceptron](https://github.com/olegkleiman/tf_keras/blob/master/perceptron_v3.py) with [GradientTape](https://www.tensorflow.org/api_docs/python/tf/GradientTape) and TF2 eager mode
6. Multi-dimentional perceptron (TBD)
7. Keras: layers of perceptrons (TBD)
8. Models: Keras.Sequential() (TBD)

## How to build
1. Use Python virtual environment (.venv) 
2. TensorFlow callbacks are served by [TensorBoard](https://www.tensorflow.org/tensorboard/get_started): $tensorboard --logdir logs/fit. Observe it at http://localhost:6000

## Notes
MNIST datasets are used in this exercises: MNIST digist - a set of handwritten 60.000 digits (28x28 px) for train set and 10.000 digits for test set, MNIST Fashion - images of clothes (same size)