# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 13:38:32 2018

@author: feiye
"""

import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32,[None,28,28,1])
y = tf.placeholder(tf.float32,[None,10])

x = tf.reshape(x,[-1,784])

w = tf.Variable(tf.random_normal([784,10]),name="w")
b = tf.Variable(tf.zeros([10]),name="b")

y_ = tf.nn.softmax(tf.matmul(x, w) + b)

cross_entropy = -tf.reduce_sum(y*tf.log(y_))

is_correct = tf.equal(tf.argmax(y_,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.005)
train_step = optimizer.minimize(cross_entropy)
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    for step in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})
        if step % 100 == 0:
            acc, loss = sess.run([accuracy, cross_entropy],feed_dict={x: batch_xs, y: batch_ys})
            print("train acc,loss:",acc,loss)
            acc, loss = sess.run([accuracy, cross_entropy],feed_dict={x: mnist.test.images, y: mnist.test.labels})
            print("test acc,loss:",acc,loss)