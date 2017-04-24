#!/usr/bin/env python
#-*- coding: utf-8 -*-

import tensorflow as tf
hello = tf.constant('HEllo ,TensorFlow')
sess = tf.Session()
print sess.run(hello)
sess.close()

# with tf.Session() as sess:
#     with tf.device("gpu:1"):
#         matrix1 = tf.constant([[3., 3.]])
#         matrix2 = tf.constant([[2.], [2.]])
#         product = tf.matmul(matrix1,matrix2)
#         result = sess.run([product])
#         print result

sess = tf.InteractiveSession()

x = tf.Variable([1.0, 2.0])
a = tf.constant([3.0, 3.0])

x.initializer.run()

sub =tf.sub(x,a)
print sub.eval()

state = tf.Variable(0, name="counter")
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)
init_op = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init_op)
    print sess.run(state)
    for _ in range(3):
        sess.run(update)
    print sess.run(state)
