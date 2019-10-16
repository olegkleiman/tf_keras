#
# This step evolves the perceptron_v1.py by introducing 
# mimimize() method on GradientDescentOptimizer
#
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

print("Devices: ", len(tf.config.experimental.list_physical_devices(device_type=None)))

x = tf.constant(1.0, name='input')
w = tf.Variable(0.8, name='weight')
y = tf.multiply(w, x, name='output')

y_ = tf.constant(0.0)
loss = tf.pow(y - y_, 2, "loss")

# minimize simply compute gradient (using tf.GradientTape) and calls apply_gragients()
train_step = tf.compat.v1.train.GradientDescentOptimizer(0.025).minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
for i in range(100):
    sess.run(train_step)
print(sess.run(y))

