# disable warnings which are just informing you if you build TensorFlow from source it can be faster on your machine.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import tensorflow.examples.tutorials.mnist import input_data
import numpy as np