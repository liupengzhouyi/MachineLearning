import torch

# method 1

class Net(torch.nn.Module):

    def __init__(self, hidden, in_features, out_features):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(in_features, hidden)
        self.out = torch.nn.Linear(hidden, out_features)

    def float(self, data):
        data = torch.relu(self.hidden(data))
        data = self.out(data)
        return data

# Building Neural network
net1 = Net(2, 10, 2)

print(net1)