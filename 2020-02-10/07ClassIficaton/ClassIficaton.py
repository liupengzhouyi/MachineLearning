import sys
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
sys.path.append("/Users/liupeng/PycharmProjects/MachineLearning/2020-02-10/07ClassIficaton/Net.py")
from Net import *

# 假数据
n_data = torch.ones(100, 2)         # 数据的基本形态
# print(n_data)
x0 = torch.normal(2 * n_data, 1)    # 类型0 x data (tensor), shape=(100, 2)
# print(x0)
y0 = torch.zeros(100)               # 类型0 y data (tensor), shape=(100, 1)
# print(y0)
x1 = torch.normal(-2 * n_data, 1)   # 类型1 x data (tensor), shape=(100, 1)
# print(x1)
y1 = torch.ones(100)                # 类型1 y data (tensor), shape=(100, 1)
# print(y1)

# 注意 x, y 数据的数据形式是一定要像下面一样 (torch.cat 是在合并数据)
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # FloatTensor = 32-bit floating
# print(x)
y = torch.cat((y0, y1), ).type(torch.LongTensor)  # LongTensor = 64-bit integer
# print(y)

# torch 只能在 Variable 上训练, 所以把它们变成 Variable
x = Variable(x)
y = Variable(y)

plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
# plt.show()

net = Net(2, 10, 2)
print(net)
# [0, 1] => 第二类
# [1, 0] => 第一类

optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
loss_func = torch.nn.CrossEntropyLoss()

for t in range(100):
    out = net(x)
    loss = loss_func(out, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if t%2==0:
        plt.cla()
        prediction = torch.max(torch.nn.functional.softmax(out),1)[1]
        pred_y = prediction.data.numpy().squeeze()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = sum(pred_y == target_y)/200
        plt.pause(0.1)

plt.ioff()
plt.show()





