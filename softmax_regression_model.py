# -*- coding:utf-8 -*-
# softmax回归模型用于手写数字图片分类
# 只有输入层和输出层，没有隐含层。是一个没有隐含层的最浅的神经网络

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# one_hot 独热编码
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True) #  mnist是一个tensorflow内部的变量
# 查看数据形式
print(mnist.train.images.shape, mnist.train.labels.shape)
print(mnist.test.images.shape, mnist.test.labels.shape)
print(mnist.validation.images.shape, mnist.validation.labels.shape)

# 创建session
sess = tf.InteractiveSession()

# 创建输入， shape=[None,784],是因为样本个数不知道，但是知道每个样本的维度是784.
x = tf.placeholder(tf.float32, shape=[None, 784])

# 创建权值和偏置 Variables
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# 实现softmax算法 Predicted Class and Cost Function
y = tf.nn.softmax(tf.matmul(x, W) + b)

# 真实标签
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# 定义loss 损失函数
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# 学习率
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# 全局参数初始化器
init = tf.initialize_all_variables()
# 初始化变量到指定值，在这种情况下都设为0
sess.run(init)

# 迭代 训练
for i in range(10000):
    if i % 100 == 0:
        print('training...', i)
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict={x:batch[0], y_: batch[1]})

# 评估
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# print accuracy
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
# close the Session
sess.close()

# 以上，构造了一个没有隐含层的最浅神经网络，搭建过程主要分为以下四点：
# (1)定义神经网络的前馈计算： y=Wx+b
# (2)定义损失函数，指定优化器，并指定优化器优化损失函数
# (3)训练：迭代地对数据进行训练
# (4)评测：验证测试集上面的准确率
# 注意：我们定义的各个公式只是计算图，并没有实际产生计算，所以需要调用run方法，并且feed数据，
# 比如cross_entropy、train_step、accuracy等都是计算节点，而不是数据结果。
