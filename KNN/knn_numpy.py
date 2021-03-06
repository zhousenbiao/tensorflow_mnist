# -*- coding:utf-8 -*-



"""
kNN(k-Nearest Neighbor)算法主要被应用于文本分类、相似推荐、面部识别、电影/音乐推荐、检测疾病等方面，
如果一个概念很难定义,但是当你看到它的时候知道它是什么,那么kNN算法就可能是合适的方法.

“k”是一个可变选项,表示任意数目的近邻.
在选定k之后,kNN算法需要一个已经分好类的训练数据集,然后对没有分类(没有标记)的记录进行分类,
kNN确定训练数据集中与该记录相似度"最近"的k条记录,将无标记的测试例子分配到k个近邻中占比最大的那个类别中.

KNN算法流程：

计算待分类点与已知类别数据集中每个点的距离；
按照距离递增次序排序；
选取与待分类点距离最小的k个点；
确定前k个点类别出现的频率；
返回前k个点出现频率最高的类别作为当前点的预测分类。

KNN的优缺点：

（1）优点：

算法简单，易于实现，不需要参数估计，不需要事先训练。

（2）缺点：

属于懒惰算法，“平时不好好学习，考试时才临阵磨枪”，意思是kNN不用事先训练，而是在输入待分类样本时才开始运行，这一特点导致kNN计算量特别大，而且训练样本必须存储在本地，内存开销也特别大。

KNN本质是基于一种数据统计的方法。KNN是一种基于记忆的学习(memory-based learning)，也是基于实例的学习(instance-based learning)，属于惰性学习(lazy learning)。
即它没有明显的前期训练过程，而是程序开始运行时，把数据集加载到内存后，不需要进行训练，就可以开始分类了。

衡量近似度的方法:
最常用的是采用欧式距离(Euclidean distance),欧式距离是通过直线距离(as the crow flies)来度量,即最短的直线路线.
另一种常见的距离度量是曼哈顿距离(anhattan distance),即两点在南北方向上的距离加上在东西方向上的距离.

-----------------------------------------------
数值计算库NumPy库总包含两种基本的数据类型：矩阵和数组，矩阵的使用类似Matlab，本实例用得多的是数组array。

1.shape()
shape是numpy函数库中的方法，用于查看矩阵或者数组的维数
# >>>shape(array) 若矩阵有m行n列，则返回(m,n)
# >>>array.shape[0] 返回矩阵的行数m，参数为1的话返回列数n

2.tile()
tile是numpy函数库中的方法，用法如下:
# >>>tile(A,(m,n))  将数组A作为元素构造出m行n列的数组

3.sum()
sum()是numpy函数库中的方法
# >>>array.sum(axis=1)按行累加，axis=0为按列累加

4.argsort()
argsort()是numpy中的方法，得到矩阵中每个元素的排序序号
# >>>A=array.argsort()  A[0]表示排序后 排在第一个的那个数在原来数组中的下标

5.dict.get(key,x)
Python中字典的方法，get(key,x)从字典中获取key对应的value，字典中没有key的话返回0

6.sorted()
python中的方法

7.min()、max()
numpy中有min()、max()方法，用法如下
# >>>array.min(0)  返回一个数组，数组中每个数都是它所在列的所有数的最小值
# >>>array.min(1)  返回一个数组，数组中每个数都是它所在行的所有数的最小值

8.listdir('str')
python的operator中的方法
# >>>strlist=listdir('str')  读取目录str下的所有文件名，返回一个字符串列表

9.split()
python中的方法，切片函数
# >>>string.split('str')以字符str为分隔符切片，返回list

"""

from numpy import *
import operator
import time
from os import listdir

"""
description:KNN算法实现分类器
parms:
    inputPoint：测试集
    dataSet：训练集
    labels：类别标签
    k：k个邻居
return:该测试数据的类别
"""
def classify(inputPoint, dataSet, labels, k):
    # 获取已知的分类的数据集（训练集）的行数，，即样本个数。shape(array) 若矩阵有m行n列，则返回(m,n)
    dataSetSize = dataSet.shape[0]
    # tile(A, (m, n)) 将数组A作为元素构造出m行n列的数组
    # tile函数将输入点拓展成与训练集相同维数的矩阵，再计算欧式距离
    # 样本与训练集的差值矩阵
    diffMat = tile(inputPoint, (dataSetSize, 1)) - dataSet

    # sqDiffMat的数据类型是nump提供的ndarray，这不是矩阵的平方，而是每个元素变成原来的平方
    sqDiffMat = diffMat ** 2
    # 计算每一行上元素的和。array.sum(axis=1)按行累加，axis=0为按列累加
    sqDistances = sqDiffMat.sum(axis=1)
    # 开方得到欧拉距离矩阵
    distances = sqDistances ** 0.5
    # 按distances中元素进行生序排序后得到的对应下标的列表，argsort函数返回的是数组值从小到大的索引值。
    sortedDistIndicies = distances.argsort()
    # classCount数据类型是这样的 {0: 2, 1: 2},字典key：value
    classCount = {}
    # 选择距离最小的k个点
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        # 类别数加1，get(key,x)从字典中获取key对应的value，没有key的话返回0
        classCount[voteIlabel] = classCount.get(voteIlabel, 0)+1
    # print(classCount)

    # 按classCount字典的第2个元素（即类别出现的次数）从大到小排序
    # sorted()函数，按照第二个元素即value的次序逆向（reverse = True）排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # print(sortedClassCount)
    return sortedClassCount[0][0]

"""
description:读取指定文件名的文本数据，构建一个矩阵
parms：
    文本文件名称
return:
    一个单行矩阵
"""
def img2vector(filename):
    returnVect = []
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect.append(int(lineStr[j]))
    return returnVect

"""
description:从文件名中解析分类数字，比如由0_0.txt得知这个文本代表的数字分类是0
parms:文本文件名称
return：一个代表分类的数字
"""
def classnumCut(fileName):
    fileStr = fileName.split('.')[0]
    classNumStr = int(fileStr.split('_')[0])
    return classNumStr

"""
description:构建训练集数据向量，以及对应分类标签向量
parms：无
return：
    hwLabels：分类标签矩阵
    trainingMat：训练数据集矩阵
"""
def trainingDataSet():
    hwLabels = []
    # 获取目录内容
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    # zeros返回全是0的矩阵，参数是行和列
    # m维向量的训练集
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        hwLabels.append(classnumCut(fileNameStr))
        trainingMat[i, :] = img2vector('trainingDigits/%s'% fileNameStr)
    return hwLabels, trainingMat

"""
description:主函数，最终打印识别了多少个数字以及识别的错误率
parms：None
return:None
"""
def handwritingTest():
    """
    hwLabels, trainingMat :标签，训练数据
    hwLabels是一个一维矩阵，代表每个文本对应的标签（即文本所代表的数字类型）
    trainingMat是一个多维矩阵，每一行都代表一个文本的数据，每行有1024个数字（0或1）
    """
    # 构建训练集
    hwLabels,trainingMat = trainingDataSet()
    # 获取测试集
    testFileList = listdir('testDigits')
    # print(testFileList)
    # 错误数
    errorCount = 0.0
    # 测试集总样本数
    mTest = len(testFileList)
    t1 = time.time()
    for i in range(mTest):
        fileNameStr = testFileList[i]
        classNumStr = classnumCut(fileNameStr)
        # img2vector返回一个文本对应的一维矩阵，1024个0或者1
        vectorUnderTest = img2vector('testDigits/%s'% fileNameStr)
        # 调用KNN算法进行测试
        classifierResult = classify(vectorUnderTest, trainingMat, hwLabels, 3)
        # 打印测试出来的结果和真正的结果，看看是否匹配
        print("the classifier came back with:%d, the real answer is: %d" % (classifierResult, classNumStr))
        # 如果测试出来的值和原值不相等，errorCount+1
        if(classifierResult != classNumStr):
            errorCount += 1.0
    # 输出测试总体样本数
    print("\n the total number of tests is: %d" % mTest)
    # 输出测试错误样本数
    print(" the total number of errors is: %d" % errorCount)
    # 输出错误率
    print(" the total number of error rate is:%f" % (errorCount/float(mTest)))
    t2 = time.time()
    # 测试耗时
    print("Cost time: %.2f min, %.4f s." % ((t2-t1)//60, (t2-t1)%60))

"""
指定handwritingTest()为主函数
"""
if __name__ == "__main__":
    handwritingTest()