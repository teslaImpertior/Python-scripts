# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 16:56:02 2018

@author: Isaac
"""

import tensorflow as tf
from datetime import datetime

with tf.device("/gpu:0"):
    # Setup operations
    print('gpu')
    d = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    e = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    startTime = datetime.now()
    f = tf.matmul(d, e)
    # Creates a session with log_device_placement set to True.
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    print(sess.run(f))
    print("Time taken GPU:", datetime.now() - startTime)
    
with tf.Session() as sess:
    # Run your code
    hello = tf.constant('Hello, TensorFlow!')
    print(sess.run(hello))
    # Creates a graph.
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    startTime = datetime.now()
    c = tf.matmul(a, b)
    # Creates a session with log_device_placement set to True.
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    # Runs the op.
    print(sess.run(c))
    print("Time taken CPU:", datetime.now() - startTime)
    print("\n" * 5)