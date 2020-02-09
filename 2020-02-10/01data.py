import torch
import numpy as np

np_data = np.arange(6).reshape((2,3))

torch_data = torch.from_numpy(np_data)

tensor2array = torch_data.numpy()

print(np_data)

print(torch_data)

print(tensor2array)





