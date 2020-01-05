
# CNN

 一张图片
 
 6个卷积内核（6个矩阵）
 
 一整图片和6个卷积内核分别滚动扫描
 
 得到6个新的图片


## Pooling

扫描图像

* 4合一
    * MaxPooling 最大值
    * AveragePooling 平均值

### 作用

* 可以做到
    * 图像降噪
    * 小范围内的特征整合得到一个新的特征
* 做不到
    * 图像旋转噪声

## 实例

* 数据类型
    * [None,784]  => [None, 28, 28,1]
    * [None, 10]


* 卷积神经网络

    * 卷积层I
        * 卷积 
            * 参数
                * filter numbers = 32个
                * flow size = [5,5]
                * strides = 1
                * padding = 'SAME'
                * bias(偏置) = 32
            * 输入[None,28,28,1]
            * 输出[None,28,28,32]
        * 激活
            * 输入[None,28,28,32]
            * function:改变数值
            * 输出[None,28,28,32]
        * 池化
            * 参数
                * size : 4(2*2)
                * strides(步长):2
                * padding:"SAME"
            * 输入：[None,28,28,32]
            * 输出：[None,14,14,32]
    * 卷积层II
        * 卷积
            * 参数
                * filter numbers = 64个
                * flow size = [5,5]
                * strides = 1
                * padding = 'SAME'
                * bias(偏置) = 64
            * 输入[None,14,14,32]
            * 输出[None,14,14,64] => ?
        * 激活
            * 输入 [None,14,14,64]
            * function:改变数值
            * 输出 [None,14,14,64]
        * 池化
            * 参数
                * size : 4(2*2)
                * strides(步长)：2
            * 输入：[None,14,14,64]
            * 输出：[None,7,7,64]
    * 全连接层
        * bias(偏置) = 32
        * 输入 [None,7,7,64]
            * 转化 [None,7*7*64]
        * 权重矩阵：[7*7*64,10]
        * 输出：[None,10]