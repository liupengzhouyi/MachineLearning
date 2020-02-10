import torch

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        # 隐藏层线性输出
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        # 输出层线性输出
        self.predict = torch.nn.Linear(n_hidden, n_output)

    # 这同时也是 Module 中的 forward 功能
    def forward(self, x):
        # 正向传播输入值, 神经网络分析出输出值
        x = torch.relu(self.hidden(x))
        # 输出值
        x = self.predict(x)
        return x