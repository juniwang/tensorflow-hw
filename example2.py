import tensorflow as tf
import numpy as np

# create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.2+ 0.4

# models
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))
y = Weights * x_data + biases

# loss
loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# train
init = tf.global_variables_initializer()

# run
sess = tf.Session()
sess.run(init) # very important

for step in range(401):
	sess.run(train)
	if step%20 == 0:
		print(step, sess.run(Weights), sess.run(biases))