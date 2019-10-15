import tensorflow as tf

# # In Tensorflow 2.0, eager execution is enabled by default.
# tf.executing_eagerly()
tf.compat.v1.disable_eager_execution()

print("Devices: ", len(tf.config.experimental.list_physical_devices(device_type=None)))
# tf.debugging.set_log_device_placement(True)

x = tf.constant(1.0, name='input')
w = tf.Variable(0.8, name='weight')
y = tf.multiply(w, x, name='output')

y_ = tf.constant(0.0)
loss = (y - y_)**2

optim =  tf.compat.v1.train.GradientDescentOptimizer(learning_rate = 0.025)

# Stochastic gradient descent
# optim = tf.keras.optimizers.SGD(learning_rate = 0.025)
# step = optim.minimize(loss, var_list = w)

grads_and_vars = optim.compute_gradients(loss)
print(grads_and_vars) # should be 1.6 after session run, 
                      # for a meanwhile prints the reference to graph's node  

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
print(sess.run(grads_and_vars[0][0]))

sess.run(optim.apply_gradients(grads_and_vars))
print(sess.run(w)) # should be 0.76
# The weight decreased by 0.04 because the optimizer subtracted 
# the gradient times the learning rate, 1.6 * 0.025, pushing the weight in the right direction.

def fit(epochs):
    calculated_weight = 0
    for step in range(epochs):
        sess.run(grads_and_vars[0][0]) # calculate new gradients
        sess.run(optim.apply_gradients(grads_and_vars))
        calculated_weight = sess.run(w) # should decrease each spep because the same resons
        
    return calculated_weight    

print(fit(100))
