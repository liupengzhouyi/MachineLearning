import torch
import numpy as np

data = [[1,23], [3, 32]]
tensor = torch.FloatTensor(data)

newDataI = np.matmul(data,data)
print(newDataI)

newDataII = torch.mm(tensor, tensor)
print(newDataII)

otherDataI = np.array(data)
newDataIII = otherDataI.dot(otherDataI)
print(newDataIII)

# newDataIV = tensor.dot(tensor)
# print(newDataIV)



