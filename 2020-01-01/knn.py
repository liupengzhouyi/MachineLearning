# knn
# test 样本找K个最接近样本
# k（100）个样本中，概率最该的样本是？（50个1）那就是1了

import tensorflow as tf
import numpy as np
import random
from tensorflow.examples.tutorials.mnist import input_data

# load date
# 数据装载
# input_data.read_data_sets('文件路径',one_hot=bool)
# one_hot=bool 非0即1
fileName = '/Users/liupeng/PycharmProjects/MachineLearning/2020-01-01/acsset'
mnist = input_data.read_data_sets(fileName,one_hot=True)

# 属性设置
# 训练集图片数量
trainNumber = 55000
# 测试集图片数量
testNumber = 10000
# 训练用的图片数量
trainSize = 5000
# 训练用的图片数量
testSize = 5

# k = ?
k = 4

# 数据分解
# 在【0-trainNumber】范围内选取trainSize个数据，不可以重复
trainIndex = np.random.choice(trainNumber,trainSize,replace=False)
#在【0-testNumber】范围内选取testSize个数据，不可以重复
testIndex = np.random.choice(testNumber,testSize,replace=False)
# 训练图片
trainData = mnist.train.images[trainIndex]
# 训练标签
trainLabel = mnist.train.labels[trainIndex]
# 测试图片
testData = mnist.test.images[testIndex]
# 测试标签
testLabel = mnist.test.labels[testIndex]

# 28*28=784
print("trainData.shape",trainData.shape) # 500*784

print("trainLabel.shape",trainLabel.shape) # 500*10

print("testData.shape",testData.shape) # 5*784

print("testLabel.shape",testLabel.shape) # 5*10

print("testLabel", testLabel) # 9 7 7 8 8

# tf input

trainDataInput = tf.placeholder(shape=[None,784], dtype=tf.float32)
trainLabelInput = tf.placeholder(shape=[None,10],dtype=tf.float32)
testDataInput = tf.placeholder(shape=[None,784], dtype=tf.float32)
testLabelInput = tf.placeholder(shape=[None,10],dtype=tf.float32)


# knn distance
# tf.expand_dim() 增加维度
# 5 * 1 * 784
# 5 500 784 （3D）2500*784
f1 = tf.expand_dims(testDataInput, 1)
# 像数差
f2 = tf.subtract(trainDataInput, f1)
# 数据累加
f3 = tf.reduce_sum(tf.abs(f2),reduction_indices=2)
# 取反
f4 = tf.negative(f3)
# 选取f4中最大的四个值
f5,f6 = tf.nn.top_k(f4,k=4)

f7 = tf.gather(trainLabelInput,f6)
# 寻求一个最大值 index
f8 = tf.reduce_sum(f7,reduction_indices=1)
# 寻求
f9 = tf.arg_max(f8,dimension=1)

with tf.Session() as sess:
    # 运行f1 参数testData 5张图片
    p1 = sess.run(f1,feed_dict={testDataInput:testData[0:5]})
    print("p1.shape = ",p1.shape)
    # p1.shape =  (5, 1, 784)
    p2 = sess.run(f2, feed_dict={trainDataInput:trainData,testDataInput:testData[0:5]})
    print("p2.shape = ", p2.shape)
    # p2.shape =  (5, 500, 784)
    p3 = sess.run(f3, feed_dict={trainDataInput:trainData,testDataInput:testData[0:5]})
    print("p3.shape = ", p3.shape)
    # p3.shape =  (5, 500)
    # 计算knn test 图片与 测试图片的距离计算
    print("p3[0,0] = ",p3[0,0])
    p4 = sess.run(f4, feed_dict={trainDataInput:trainData,testDataInput:testData[0:5]})
    print("p4.shape = ", p4.shape)
    print("p4[0,0]=",p4[0,0])
    p5,p6 = sess.run((f5,f6),feed_dict={trainDataInput:trainData,testDataInput:testData[0:5]})
    print("p5.shape", p5.shape)
    print("p6.shape", p6.shape)
    print("差值:",p5)
    print("坐标:",p6)
    p7 = sess.run(f7, feed_dict={trainDataInput:trainData,testDataInput:testData[0:5],trainLabelInput:trainLabel})
    print("p7.shape = ", p7.shape)
    # p7.shape =  (5, 4, 10)
    print(p7)
    p8 = sess.run(f8, feed_dict={trainDataInput:trainData,testDataInput:testData[0:5],trainLabelInput:trainLabel})
    print("p8.shape = ", p8.shape)
    print(p8)
    p9 = sess.run(f9, feed_dict={trainDataInput:trainData,testDataInput:testData[0:5],trainLabelInput:trainLabel})
    print("p9.shape = ", p9.shape)
    print(p9)

