# -*- coding:utf-8 -*-
# tf.nn.softmax_cross_entropy_with_logits的用法
import tensorflow as tf

#our NN's output
logits = tf.constant([[1.0,2.0,3.0],[1.0,2.0,3.0],[1.0,2.0,3.0]])
#step1:do softmax
y = tf.nn.softmax(logits)
#true label
y_ = tf.constant([[0.0,0.0,1.0],[0.0,0.0,1.0],[0.0,0.0,1.0]])
#step2:do cross_entropy
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
#do cross_entropy just one step
cross_entropy2 = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels= y_))#dont forget tf.reduce_sum()!!

with tf.Session() as sess:
    softmax = sess.run(y)
    c_e = sess.run(cross_entropy)
    c_e2 = sess.run(cross_entropy2)
    print("step1:softmax result=")
    print(softmax)
    print("step2:cross_entropy result=")
    print(c_e)
    print("Function(softmax_cross_entropy_with_logits) result=")
    print(c_e2)

    # 注意！！！这个函数的返回值并不是一个数，而是一个向量，
    # 如果要求交叉熵，我们要再做一步tf.reduce_sum操作, 就是对向量里面所有元素求和，
    # 最后才得到，如果求loss，则要做一步tf.reduce_mean操作，对向量求均值！


