# cnn: Convolutional Neural Network

# disable warnings which are just informing you if you build TensorFlow from source it can be faster on your machine.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# number 1 to 10 data
mnist = input_data.read_data_sets('MINST_data', one_hot=True)

def compute_accuracy(v_xs, v_ys):
	global prediction
	y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob:1})
	correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob:1})
	return result


def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	# stride [1, x_movement, y_movement, 1]
	# must have strides[0] = strides[3] = 1
	# padding could be 'SAME'(keep the same size) or 'VALID'(the size is smaller than original)
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	# two pooling methods: max_pool or averge_pool
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 784]) # 28*28, the size of pic
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32) # define how many data won't be dropped out
x_image = tf.reshape(xs, [-1, 28, 28, 1])
print(x_image.shape) # [n_samples, 28, 28 ,1]. The last `1` means channel. 


# conv1 layer
W_conv1 = weight_variable([5,5,1,32]) # patch 5x5, in height 1, out height 32
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # output image size 28x28x32 because strides [1,1,1,1] and padding is 'SAME'
h_pool1 = max_pool_2x2(h_conv1) # output image size 14x14x32 because strides is [1,2,2,1]

# conv2 layer
W_conv2 = weight_variable([5,5,32,64]) # patch 5x5, in height 32, out height 64
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output image size 14x14x64
h_pool2 = max_pool_2x2(h_conv2) # output image size 7x7x64

# func1 layer, full connected neural network
W_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])
# convert [n_samples,7,7,64] to [n_samples, 7*7*64], 3D=>1D
h_pool2_flat = tf.reshape(h_pool2, [-1,7*7*64]) 
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob) # dropout to prevent over fitting

# func2 layer, full connected neural network
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1])) # loss
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # very important!!!

for i in range(1000):
	# trainng
	batch_xs, batch_ys = mnist.train.next_batch(100)  # extract 100 data from minist. Not load all data for quick training
	sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob:0.5})
	if i%50 == 0 :
		# to see the accuracy
		print(compute_accuracy(mnist.test.images[:1000], mnist.test.labels[:1000]))