import torch
import numpy as np

data = [-2,-4, 6,- 8]

# 32浮点数
tensor = torch.FloatTensor(data)
print(tensor)


# abs
absData = np.abs(data)
print(absData)
# [2 4 6 8]

