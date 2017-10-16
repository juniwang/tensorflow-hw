# disable warnings which are just informing you if you build TensorFlow from source it can be faster on your machine.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = tf.multiply(input1, input2)

with tf.Session() as sess:
	print(sess.run(output, feed_dict={input1:[7.], input2:[2.3]}))