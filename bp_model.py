# -*-coding:utf-8 -*-
# BP神经网络实现手写数字识别，来自https://www.cnblogs.com/bigmoyan/archive/2015/05/22/4523255.html

import math
import numpy as np
import scipy.io as sio


# 读入数据
################################################################################################
print "输入样本文件名（需放在程序目录下）"
filename = './MNIST_mat/mnist_train.mat'     # raw_input() # 换成raw_input()可自由输入文件名
sample = sio.loadmat(filename)
sample = sample["mnist_train"]
sample /= 256.0       # 特征向量归一化

print "输入标签文件名（需放在程序目录下）"
filename = './MNIST_mat/mnist_train_labels.mat'   # raw_input() # 换成raw_input()可自由输入文件名
label = sio.loadmat(filename)
label = label["mnist_train_labels"]

##################################################################################################


# 神经网络配置
##################################################################################################
samp_num = len(sample)      # 样本总数
inp_num = len(sample[0])    # 输入层节点数
out_num = 10                # 输出节点数
hid_num = 6  # 隐层节点数(经验公式)
w1 = 0.2*np.random.random((inp_num, hid_num))- 0.1   # 初始化输入层权矩阵
w2 = 0.2*np.random.random((hid_num, out_num))- 0.1   # 初始化隐层权矩阵
hid_offset = np.zeros(hid_num)     # 隐层偏置向量
out_offset = np.zeros(out_num)     # 输出层偏置向量
inp_lrate = 0.3             # 输入层权值学习率
hid_lrate = 0.3             # 隐层学权值习率
err_th = 0.01                # 学习误差门限


###################################################################################################

# 必要函数定义
###################################################################################################
def get_act(x):
    act_vec = []
    for i in x:
        act_vec.append(1/(1+math.exp(-i)))
    act_vec = np.array(act_vec)
    return act_vec

def get_err(e):
    return 0.5*np.dot(e,e)


###################################################################################################

# 训练——可使用err_th与get_err() 配合，提前结束训练过程
###################################################################################################

for count in range(0, samp_num):
    print count
    t_label = np.zeros(out_num)
    t_label[label[count]] = 1
    #前向过程
    hid_value = np.dot(sample[count], w1) + hid_offset       # 隐层值
    hid_act = get_act(hid_value)                # 隐层激活值
    out_value = np.dot(hid_act, w2) + out_offset             # 输出层值
    out_act = get_act(out_value)                # 输出层激活值

    #后向过程
    e = t_label - out_act                          # 输出值与真值间的误差
    out_delta = e * out_act * (1-out_act)                                       # 输出层delta计算
    hid_delta = hid_act * (1-hid_act) * np.dot(w2, out_delta)                   # 隐层delta计算
    for i in range(0, out_num):
        w2[:,i] += hid_lrate * out_delta[i] * hid_act   # 更新隐层到输出层权向量
    for i in range(0, hid_num):
        w1[:,i] += inp_lrate * hid_delta[i] * sample[count]      # 更新输出层到隐层的权向量

    out_offset += hid_lrate * out_delta                             # 输出层偏置更新
    hid_offset += inp_lrate * hid_delta

###################################################################################################

# 测试网络
###################################################################################################
filename = './MNIST_mat/mnist_test.mat'  # raw_input() # 换成raw_input()可自由输入文件名
test = sio.loadmat(filename)
test_s = test["mnist_test"]
test_s /= 256.0

filename = './MNIST_mat/mnist_test_labels.mat'  # raw_input() # 换成raw_input()可自由输入文件名
testlabel = sio.loadmat(filename)
test_l = testlabel["mnist_test_labels"]
right = np.zeros(10)
numbers = np.zeros(10)
                                    # 以上读入测试数据
# 统计测试数据中各个数字的数目
for i in test_l:
    numbers[i] += 1

for count in range(len(test_s)):
    hid_value = np.dot(test_s[count], w1) + hid_offset       # 隐层值
    hid_act = get_act(hid_value)                # 隐层激活值
    out_value = np.dot(hid_act, w2) + out_offset             # 输出层值
    out_act = get_act(out_value)                # 输出层激活值
    if np.argmax(out_act) == test_l[count]:
        right[test_l[count]] += 1
print right
print numbers
result = right/numbers
sum = right.sum()
print result
print sum/len(test_s)
###################################################################################################
# 输出网络
###################################################################################################
Network = open("MyNetWork", 'w')
Network.write(str(inp_num))
Network.write('\n')
Network.write(str(hid_num))
Network.write('\n')
Network.write(str(out_num))
Network.write('\n')
for i in w1:
    for j in i:
        Network.write(str(j))
        Network.write(' ')
    Network.write('\n')
Network.write('\n')

for i in w2:
    for j in i:
        Network.write(str(j))
        Network.write(' ')
Network.write('\n')

Network.close()


# BP神经网络运行机理：
# 一个特征向量的各个分量按不同权重加权，再加一个常数项偏置，生成隐层各节点的值。
# 隐层节点的值经过一个激活函数激活后，获得隐层激活值，隐层激活值仿照输入层到隐层的过程，加权再加偏置，获得输出层值，
# 通过一个激活函数得到最后的输出。
# 以上过程是前向过程，是信息的流动过程。各层之间采用全连接方式，权向量初始化为随机向量，激活函数可使用值域为（0,1）的
# sigmoid函数，也可以使用值域为（-1,1）的tanh函数，两个函数的求导都很方便。
# BP神经网络的核心数据是权向量，经过初始化后，需要在训练数据的作用下一次次迭代更新权向量，直到权向量能够正确表达输入输出的映射关系为止。
# 权向量的更新是根据预测输出与真实值的差来更新的，前面说过BP神经网络是一个有监督学习模型，对于一个特征向量，可以通过神经网络前向过程得到一个预测输出，
# 而该特征向量的label又是已知的，二者之差就能表达预测与真实的偏差情况。这就是后向过程，后向过程是误差流动的过程。抛开具体的理论推导不谈，从编程来说，
# 权值更新的后向过程分为两步。
# 第一步，为每个神经元计算偏差δ，δ是从后向前计算的，故称之为后向算法。对输出层来说，偏差是 act（预测值-真值），act为激活函数。对隐层而言，
# 需要从输出层开始，逐层通过权值计算得到。确切的说，例如对上面的单隐层神经网络而言，隐层各神经元的δ就是输出层δ乘以对应的权值，如果输出层有多个神经元，
# 则是各输出层神经元δ按连接的权值加权生成隐层的神经元δ。
# 第二步，更新权值，w=w+η*δ*v 其中η是学习率，是常数。v是要更新的权值输入端的数值，δ是要更新的权值输出端的数值。例如更新隐层第一个神经元到输出层的权值，
# 则δ是第一步计算得到的输出层数值，v是该权值输入端的值，即隐层第一个神经元的激活值。同理如果更新输入层第一个单元到隐层第一个单元的权值，
# δ就是隐层第一个单元的值，而v是输入层第一个单元的值。偏置可以看作“1”对各神经元加权产生的向量，因而其更新方式相当于v=1的更新，不再赘述。
# 在编程中可以将1强行加入各层输入向量的末尾，从而不单独进行偏置更新，也可以不这样做，单独把偏置抽出来更新。以上的算法采用的是第二种方法。

# 程序结构：
# 程序分四个部分，第一个部分数据读取，第二个部分是神经网络的配置，第三部分是神经网络的训练，第四部分是神经网络的测试，最后还有个神经网络的保存

