# tensorflow_mnist
基于tensorflow用各种不同的方法来实现mnist手写数字图片分类
MNIST是一个包含数字0~9的手写体图片数据集，图片已归一化为以手写数 字为中心的28*28规格的图片。MNIST由训练集与测试集两个部分组成。
MNIST数据集的手写数字样例:
 MNIST数据集中的每一个图片由28*28个像素点组成
 每个像素点的值区间为0~255，
0表示白色，255表示黑色。

- 1.softmax回归模型用于手写数字图片分类 softmax_regression_model.py
- 2.单层softmax模型用于手写数字图片识别 onelayer_softmax_regression_model.py
- 3.BP神经网络模型实现手写数字图片识别 bp_model.py
