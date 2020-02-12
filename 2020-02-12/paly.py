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
DOWNLOAD_MNIST = True  # 如果你已经下载好了mnist数据就写上 Fasle

train_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=True,
    transform=transforms.ToTensor(),
    download=DOWNLOAD_MNIST,
)

# 批训练 50samples, 1 channel, 28x28 (50, 1, 28, 28)
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)









