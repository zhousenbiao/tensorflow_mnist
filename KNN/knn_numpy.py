# -*- coding:utf-8 -*-

# KNN算法流程
#
# 计算待分类点与已知类别数据集中每个点的距离；
# 按照距离递增次序排序；
# 选取与待分类点距离最小的k个点；
# 确定前k个点类别出现的频率；
# 返回前k个点出现频率最高的类别作为当前点的预测分类。

from numpy import *

# KNN算法实现分类器
def classify(inputPoint, dataSet, labels, k):
    # 获取已知的分类的数据集（训练集）的行数
    dataSetSize = dataSet.shape[0]
    # tile函数讲输入点拓展成与训练集相同维数的矩阵，再计算欧式距离
    # 样本与训练集的差值矩阵
    diffMat = tile(inputPoint, (dataSetSize, 1)) - dataSet
    #
    sqDiffMat = diffMat ** 2
    # 计算每一行上元素的和
    sqDistances = sqDiffMat.sum(axis=1)
    # 开方得到欧拉距离矩阵
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    #
    classCount = {}
    # 选择距离最小的k个点
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        # 类别数加1
        classCount[voteIlabel] = classCount.get(voteIlabel, 0)+1
    print(classCount)

    sortedClassCount = sorted(classCount.items(), key= operator.itemgetter(1), reverse= True)
    print(sortedClassCount)
    return sortedClassCount[0][0]