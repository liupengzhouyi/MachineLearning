import torch
import torch.utils.data as Data
from torch.autograd import  Variable
import matplotlib.pyplot as plt

BATCH_SIZE = 32
EPOCH = 12
LR = 0.01

x = torch.unsqueeze(torch.linspace(-1, 1, 400), dim=1)
y = x.pow(2) + 0.1*torch.normal(torch.zeros(*x.size()))
# print(x)
# print(y)

# plot dataset
# plt.scatter(x.numpy(), y.numpy())
# plt.show()

torch_dataset = Data.TensorDataset(Variable(x), Variable(y))
loader = Data.DataLoader(
    # torch TensorDataSet format
    dataset=torch_dataset,
    # mini batch size
    batch_size=BATCH_SIZE,
    # 是否乱序
    shuffle=True,
    # 多线程 读数
    num_workers = 2,
)


# 默认的 network 形式
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(1, 20)  # hidden layer
        self.predict = torch.nn.Linear(20, 1)  # output layer

    def forward(self, x):
        x = torch.relu(self.hidden(x))  # activation function for hidden layer
        x = self.predict(x)  # linear output
        return x


# 为每个优化器创建一个 net
net_SGD = Net()
net_Adam = Net()
nets = [net_SGD, net_Adam]

if __name__ =='__main__':

    opt_SGD = torch.optim.SGD(net_SGD.parameters(), lr=LR)
    opt_Adam = torch.optim.Adam(net_Adam.parameters(), lr=LR, betas=(0.9, 0.99))
    optimizers = [opt_SGD, opt_Adam]

    loss_func = torch.nn.MSELoss()
    losses_his = [[], []]

    for epoch in range(EPOCH):
        for step, (batch_x, batch_y) in enumerate(loader):
            for net, opt, l_his in zip(nets, optimizers, losses_his):
                output = net(batch_x)
                loss = loss_func(output, batch_y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                l_his.append(loss.data)

    libels = ['SGD', 'Adam']

    for i, l_his in enumerate(losses_his):
        plt.plot(l_his, label=libels[i])

    plt.legend(loc="best")
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.ylim((0, 0.2))
    plt.show()

# opitmizer = torch.optim.SGD()
