# -*-coding:utf-8 -*-
# CNN卷积神经网络实现手写数字图片识别
# 环境：python3+
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)


# 权重
# 权重不能初始化为全部相同的值，要将参数进行随机初始化，而不是全部置为0。如果所有参数都用相同的值作为初始值，
# 那么所有隐藏层单元最终会得到与输入值有关的、相同的函数（也就是说同一层的所有结点都会有相同的激活函数）。
# 随机初始化的目的是为了使对称失效（symmetry breaking）。
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# 偏置
# 同时在tensorflow的深度网络里面，他们使用了ReLU(Rectified Linear Unites）作为激活函数，
# 为了避免出现神经元不能被激活（dead neurons）的情况，在这次的训练网络中又加入了一些偏置量。
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# 卷积
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# 池化，2x2最大池化，即将一个2x2的像素块降为1x1的像素。最大池化会保留原始像素块中灰度值最高的那一个像素，
# 即保留最显著的特征
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# x是特征，y是真实的label，28*28的灰度图，像素个数784
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# reshape x，对图像做预处理，将1D的输入向量转化为2D的图片结构，即1*784到28*28的结构，
# -1代表样本数量不固定，1代表颜色通道数量
x_image = tf.reshape(x, [-1, 28, 28, 1])

# session
sess = tf.InteractiveSession()
# 第一层
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
# 卷积和池化
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 第二层
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
# 卷积和池化
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 全连接层，将两次池化后的7*7共128个特征图转化为1D向量，隐含节点1024由自己定义
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout层，为了减轻过拟合
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# readout 层，dropout层输出连接一个Softmax层，得到最后的概率输出
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
# 前向传播的预测值
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

print("CNN ready!")

# 定义损失函数为交叉熵损失函数
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y_conv), reduction_indices=[1]))
# 优化器
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# 定义评测准确率的操作
# 对比预测值的索引和真实值label的索引是否一样，一样返回True，不一样返回False
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# 初始化所有参数 initialize variables
sess.run(tf.initialize_all_variables())

print("Functions Ready!")

# 所有样本迭代1000次
training_epochs = 1000
# 每进行一次迭代选择100个样本
batch_size = 100
display_step = 1

# 训练
for i in range(training_epochs):
    batch = mnist.train.next_batch(batch_size)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print('训练步骤 %d, accuracy是: %g' % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

# 测试
test_accuracy = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})

print('测试的accuracy是: %g' % test_accuracy)
