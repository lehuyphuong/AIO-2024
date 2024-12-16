import torch
import torch.nn as nn
import numpy as np

data = np.array([[[1, 6], [3, 4]]])
data = torch.tensor(data, dtype=torch.float32)

bnorm = nn.BatchNorm2d(1)
data = data.unsqueeze(0)
with torch.no_grad():
    output = bnorm(data)
    print(output)
