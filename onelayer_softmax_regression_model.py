# -*- coding:utf-8 -*-
# softmax回归模型用于手写数字图片分类

# 定义了一个添加层函数add_layer()

import tensorflow as tf
# 下载mnist数据
from tensorflow.examples.tutorials.mnist import input_data
# 数字1-10
mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)
# 添加层函数
def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1,)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b,)
    return outputs

# 计算精确度函数
def compute_accuracy(v_xs, v_ys):
    # 把预测prediction作为全局变量
    global prediction
    # 测试数据
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    # 比较真实标签和预测标签
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    # 计算预测正确百分百
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # 激活
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result

# 传入输入值 传入28*28的图片
xs = tf.placeholder(tf.float32, [None, 784])
# 传入图片标签值
ys = tf.placeholder(tf.float32, [None, 10])
# 输出层
prediction = add_layer(xs, 784, 10, activation_function=tf.nn.softmax)
# 误差，此处计算误差方法与之前的方法不同，一般在分类中使用
cross_entropy = tf.reduce_mean(tf.reduce_sum(ys*tf.log(prediction), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
    if i % 50 == 0:
        print(compute_accuracy(mnist.test.images, mnist.test.labels))