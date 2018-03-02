# -*- coding:utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets("../MNIST_data",one_hot=True)
import os
def convolutional(x, keep_prob):
    def conv2d(x, w):
        return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="SAME")

    def max_pool(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    x_image = tf.reshape(x, [-1, 28, 28, 1])
    w_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1)+ b_conv1)
    h_pool1 = max_pool(h_conv1)
    print("搭建了网络的第一层")

    w_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
    h_pool2 = max_pool(h_conv2)
    print("搭建了网络的第二层")

    w_fc1 = weight_variable([7*7*64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1)+b_fc1)
    print("搭建了网络的第三层")

    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    w_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

    return y, [w_conv1, b_conv1, w_conv2, b_conv2, w_fc1, b_fc1, w_fc2, b_fc2]

# with tf.VariableScope("convolutional"):
x = tf.placeholder(tf.float32, [None, 784])
keep_prob = tf.placeholder(tf.float32)
y, variables = convolutional(x, keep_prob)

y_ = tf.placeholder(tf.float32, [None, 10])
# 定义loss/cost function
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
# 创建accuracy op 和 train_step op操作，用于在sess会话中进行运行
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver(variables)
# 初始化所有变量
init_op = tf.global_variables_initializer()

# 启动session个会话
with tf.Session() as sess:
    sess.run(init_op)
    for i in range(200):
        batch = data.train.next_batch(50)
        if i % 100 == 0:
            # 在tensorflow中，在一个With tf.Session() as sess底下执行一个op操作执行eval()函数等价与执行sess.run(op)操作
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))
        sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    print(sess.run(accuracy, feed_dict={x: data.test.images, y_: data.test.labels, keep_prob: 1.0}))

    path = saver.save(sess, os.path.join(os.path.dirname(__file__), 'data', 'convolutional.ckpt'), write_meta_graph=False, write_state=False)
    print("Saved:", path)