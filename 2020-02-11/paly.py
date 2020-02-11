import sys

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import matplotlib.pyplot as plt
import torchvision
sys.path.append("/Users/liupeng/PycharmProjects/MachineLearning/2020-02-11/CNN.py")
from CNN import *


EPOCH = 1
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = False

train_data = torchvision.datasets.MNIST(
    root="./mnist",
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST,
)

# plot one example
print(train_data.train_data.size())
print(train_data.test_labels.size())
# plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
# plt.title('%i' % train_data.train_labels[0])
# plt.show()


train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    # num_workers=2
)

test_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=False
)
test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1), volatile=True).type(torch.FloatTensor)[:2000]/255.
test_y = test_data.test_labels[:2000]

myCNN = CNN()

print(myCNN)

optimizer = torch.optim.Adam(myCNN.parameters(), lr=LR)
loss_func = torch.nn.CrossEntropyLoss()


for epoch in range(BATCH_SIZE):
    for step, (x, y) in enumerate(train_loader):
        b_x = Variable(x)
        b_y = Variable(y)
        output = myCNN(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step%50 == 0:
            test_output = myCNN(test_x)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = float((pred_y == test_y.data).sum()) / float(test_y.size(0))
            print('Epoch: ', epoch, "| train loss: %.4f" % loss.data, '| test accuracy: ', accuracy)


test_output = myCNN(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(pred_y, 'prediction number')
print(test_y[:10].numpy(), 'real number')

