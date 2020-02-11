import torch
from torch.autograd import Variable
import torch.utils.data as Data
import matplotlib.pyplot as plt
import torchvision

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Sequential(
            # 卷积 1 , 28 , 28
            torch.nn.Conv2d(
                # 输入图片高度
                in_channels=1,
                # 输出高度
                out_channels=16,
                # 建立一个5*5的扫描仪
                kernel_size=5,
                # 步长
                stride=1,
                # 包？几圈0
                padding=2,
            ), # 16 * 28 * 28
            # 损失函数
            torch.nn.ReLU(), # 16 , 28 , 28
            # 池化
            torch.nn.MaxPool2d(
                # 建立一个2*2的扫描仪
                kernel_size=2,
            ), # 16 , 14 , 14
        )
        self.conv2 = torch.nn.Sequential(       # 16 , 14 , 14
            torch.nn.Conv2d(16, 32, 5, 1, 2),   # 32 , 14 , 14
            torch.nn.ReLU(),                    # 32 , 14 , 14
            torch.nn.MaxPool2d(2)               # 32 , 7 , 7
        )
        self.out = torch.nn.Linear(32 * 7 * 7, 10)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)           # (biach, 32, 7, 7)
        x = x.view(x.size(0), -1)   # (biach, 32 * 7 * 7)
        output = self.out(x)
        return output
