import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy

# create data
x = torch.unsqueeze(torch.linspace(-1, 1, 500), dim=1)      # shape: torch.Size([100, 1])
y = x.pow(2) + 0.2*torch.rand(x.size())
# requires_grad=True   要求梯度
# requires_grad=False   不要求梯度
x = Variable(x, requires_grad=False)
y = Variable(y, requires_grad=False)


def save():
    net1 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )
    optimizer = torch.optim.SGD(net1.parameters(), lr=0.5)
    loss_func = torch.nn.MSELoss()

    for t in range(500):
        prediction = net1(x)
        loss = loss_func(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # plot result
    plt.figure(1, figsize=(5, 13))
    plt.subplot(311)
    plt.title("net-I")
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), '-r', lw=5)

    # 保存整个神经网络
    torch.save(net1, 'net.pkl')  # entire net
    # 保存整个神经网络的参数
    torch.save(net1.state_dict(), 'net_params.pkl')  # parameters

def restore_net():
    net2 = torch.load('net.pkl')
    prediction = net2(x)
    # polt result
    plt.subplot(312)
    plt.title('net-II')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)


def restore_params():
    net3 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )
    net3.load_state_dict(torch.load("net_params.pkl"))
    prediction = net3(x)
    plt.subplot(313)
    plt.title('net-III')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.scatter(x.data.numpy(), prediction.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
    plt.show()


save()
restore_net()
restore_params()
