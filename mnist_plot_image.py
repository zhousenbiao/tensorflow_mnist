# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

data = input_data.read_data_sets("./MNIST_data", one_hot=True)
def plot_images(images,cls_true,cls_pred=None):
    assert len(images)==len(cls_true)==9

    #在3*3的栅格中画出9张图像，然后在每张图像下面写出真实的类别和预测类别
    fig,axes=plt.subplots(3,3)
    fig.subplots_adjust(hspace=0.3,wspace=0.3)

    for i,ax in enumerate(axes.flat):
        #开始制作图片
        ax.imshow(images[i].reshape([28,28]),cmap="binary")

        #展示正确的和预估的类别
        if cls_pred is None:
            xlabel="True:{0}".format(np.where(cls_true[i]==1)[0])
        else:
            xlabel="True:{0},Pred:{1}".format(cls_true[i],cls_pred[i])

        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()

images = data.test.images[0:9]
cls_true = data.test.labels[0:9]

plot_images(images=images, cls_true=cls_true)