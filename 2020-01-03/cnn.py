
import tensorflow as tf
import numpy as np
import random
from tensorflow.examples.tutorials.mnist import input_data

print("----begin---")

def weight_variables(shape):
    """
    a function to defining init weight variables
    :param shape:
    :return:
    """
    w = tf.Variable(tf.random_normal(shape=shape, mean=0.0,stddev=1.0))
    return w


def bias_variables(shape):
    """
    a function to defining init bias variables
    :param shape:
    :return:
    """
    b = tf.Variable(tf.constant(0.0, shape=shape))
    return b

def model():
    """
    defining convlution model
    :return:
    """
    # 1, 准备数据占位符 x [None, 784] y_true [None, 10]
    with tf.variable_scope('data'):
        x = tf.placeholder(tf.float32, [None, 784])
        y_ture = tf.placeholder(tf.int32, [None,10])

    # convlution I
    with tf.variable_scope("conv1"):
        # 改变形状[None, 784] -> [-1, 28,28,1]
        x_reshape = tf.reshape(x, [-1,28,28,1])
        # 初始化权重
        w_conv1 = weight_variables([5,5,1,32])
        # 初始化偏置
        b_conv1 = bias_variables([32])
        # 卷积
        # tf.nn.conv2d(input=, filter=, strides=, padding=, name=None)
        # input = 输入张量,
        # filter = 过滤器的大小,
        # strides = [1, stride, stride, 1], 步长
        # padding = 填充算法"SAME"(超出部分填充),"VALID"(超出部分舍弃),
        # !* name = None
        temp = tf.nn.conv2d(input=x_reshape, filter=w_conv1, strides=[1,1,1,1], padding='SAME') + b_conv1
        # 激活
        x_relu1 = tf.nn.relu(temp)
        # 池化 : [None,28,28,32] -> [None,14,14,32]
        # value: 4-D [batch, height. width, channels]
        # ksize: 池化窗口大小 [1,ksize,ksize,1]
        # strides: 步长大小 [1,strides,strides,1]
        # padding = 填充算法"SAME"(超出部分填充),"VALID"(超出部分舍弃),
        tf.nn.max_pool(value=x_relu1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    # convlution II
    with tf.variable_scope("conv2"):

    return None

def conv_fc():
    # setting data path
    fileName = './2020-01-03/acsset'
    # get data
    mnist = input_data.read_data_sets(fileName,one_hot=True)

    # defining model and get conclusion
    model()

    return None
