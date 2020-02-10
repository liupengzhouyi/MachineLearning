import sys

import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
sys.path.append("/Users/liupeng/PycharmProjects/MachineLearning/2020-02-10/06Regression/Net.py")
from Net import *

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = x.pow(2) + 0.2 * torch.rand(x.size())

x = Variable(x)
y = Variable(y)

net = Net(1, 10, 1)
print(net)

plt.ion()
plt.show()

optimizer = torch.optim.SGD(net.parameters(), lr=0.5)

loss_func = torch.nn.MSELoss()

for t in range(100):
    prediction = net(x)
    # 计算误差
    loss = loss_func(prediction, y)
    # 归0
    optimizer.zero_grad()
    # 反向传递
    loss.backward()
    # 将参数更新值施加到 net 的 parameters 上
    optimizer.step()

    # 接着上面来
    if t % 5 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        # plt.text(0.5, 0, 'Loss=%.4f' % loss.data[0], fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()