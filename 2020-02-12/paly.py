import torch
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data
import matplotlib.pyplot as plt

torch.manual_seed(1)  # reproducible

# Hyper Parameters
EPOCH = 1  # 训练整批数据多少次, 为了节约时间, 我们只训练一次
BATCH_SIZE = 64
TIME_STEP = 28  # rnn 时间步数 / 图片高度
INPUT_SIZE = 28  # rnn 每步输入值 / 图片每行像素
LR = 0.01  # learning rate
DOWNLOAD_MNIST = False  # 如果你已经下载好了mnist数据就写上 Fasle

train_data = torchvision.datasets.MNIST(root='./mnist', train=True, transform=transforms.ToTensor(), download=DOWNLOAD_MNIST,)
# 批训练 50samples, 1 channel, 28x28 (50, 1, 28, 28)
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_data = torchvision.datasets.MNIST(root='./mnist', train=False, transform=transforms.ToTensor(),)
# 为了节约时间, 我们测试时只测试前2000个
test_x = Variable(test_data.test_data).type(torch.FloatTensor)[:200]/255.  # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
test_y = test_data.test_labels.numpy().squeeze()[:200]


class RNN(torch.nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.run = torch.nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )
        self.out = torch.nn.Linear(
            64, 18
        )

    def forward(self, x):
        r_out, (h_n, h_c) = self.run(x, None)
        out = self.out(r_out[:, -1, :])
        return out

rnn = RNN()
print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)  # optimize all parameters
loss_func = torch.nn.CrossEntropyLoss()  # the target label is not one-hotted

# training and testing
for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):  # gives batch data
        b_x = Variable(x.view(-1, 28, 28))  # reshape x to (batch, time_step, input_size)
        b_y = Variable(y)  # batch y

        output = rnn(b_x)  # rnn output
        loss = loss_func(output, b_y)  # cross entropy loss
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # back propagation, compute gradients
        optimizer.step()  # apply gradients

        if step%50 == 0:
            test_output = rnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
            accuracy = sum(pred_y == test_y) / test_y.size
            print('Epoch: ', epoch, "| train loss: %.4f" % loss.data, '| test accuracy: ', accuracy)

test_output = rnn(test_x[:10].view(-1, 28, 28))
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(pred_y, 'prediction number')
print(test_y[:10], 'real number')


