
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
        temp1 = tf.nn.conv2d(input=x_reshape, filter=w_conv1, strides=[1,1,1,1], padding='SAME') + b_conv1
        # 激活
        x_relu1 = tf.nn.relu(temp1)
        # 池化 : [None,28,28,32] -> [None,14,14,32]
        # value: 4-D [batch, height. width, channels]
        # ksize: 池化窗口大小 [1,ksize,ksize,1]
        # strides: 步长大小 [1,strides,strides,1]
        # padding = 填充算法"SAME"(超出部分填充),"VALID"(超出部分舍弃),
        x_pool1 = tf.nn.max_pool(value=x_relu1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    # convlution II
    # 卷积核size : 5*5*32
    # 偏置 ： 64
    # 权重： [5, 5, 32, 64]
    with tf.variable_scope("conv2"):
        # 初始化权重
        w_conv2 = weight_variables([5,5,32,64])
        #初始化偏置
        b_conv2 = bias_variables([64])
        # 卷积
        temp2 = tf.nn.conv2d(input=x_pool1, filter=w_conv2, strides=[1,1,1,1], padding='SAME') + b_conv2
        # 激活
        x_relu2 = tf.nn.relu(temp2)
        # 池化 [None, 14, 14, 64] -> [None, 7, 7, 64]
        x_pool2 = tf.nn.max_pool(value=x_relu2, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

    with tf.variable_scope():
        # 随机初始化权重
        w_fc = weight_variables([7*7*64, 10])
        # 随机初始化偏置
        b_fc = bias_variables([10])
        # 修改数据形状
        x_fc_reshape = tf.reshape(x_pool2, [-1,7*7*64])
        # 进行矩阵运算，得出每一个样本的10个结果
        y_predict = tf.matmul(x_fc_reshape, w_fc) + b_fc

    return x, y_ture, y_predict

def conv_fc():
    # setting data path
    fileName = './2020-01-03/acsset'
    # get data
    mnist = input_data.read_data_sets(fileName,one_hot=True)

    # defining model and get conclusion
    x, y_ture, y_predict = model()

    # 进行交叉熵运算计算损失
    with tf.variable_scope("sotf_cross"):
        # 求平均交叉熵损失
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_ture, logits=y_predict))

    # 梯度下降，求出损失
    with tf.variable_scope("optimizer"):
        train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    # 计算准确率
    with tf.variable_scope("acc"):
        equal_list = tf.equal(tf.argmax(y_ture, 1), tf.argmax(y_predict, 1))
        accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))

    # 定义一个初始化变量 op
    init_op = tf.global_variables_initializer()

    # 开启会话运行
    with tf.Session() as sess:
        sess.run(init_op)

        # 循环训练
        for i in range(100):

            # 取出真实的数据
            mnist_x, mnist_y = mnist.train.next_batch(50)

            # run train_op 训练
            sess.run(train_op, feed_dict={x:mnist_x, y_ture: mnist_y})

            print("训练第%d部, 准确率为%f" % (i, sess.run(accuracy, feed_dict={x:mnist_x, y_ture: mnist_y})))



if __name__ == "__main__":
    conv_fc()

