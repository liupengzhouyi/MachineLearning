import torch
from torch.autograd import Variable
import torch.utils.data as Data

BATCH_SIZE = 8

x = torch.linspace(1, 10, 10)
y = torch.linspace(10, 1, 10)

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

for epoch in range(3):
    for step, (batch_x, batch_y) in enumerate(loader):
        # training....
        print('Epoch: ', epoch, '| Step: ', step, '| batch_x: ', batch_x, '| batch_y: ', batch_y)
