import torch
import torch.nn as nn

seed = 1
torch.manual_seed(seed)
input_tensor = torch.Tensor([[[[1.0, 2.0], [3.0, 4.0]]]])

conv_layer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1)
conv_output = conv_layer(input_tensor)

with torch.no_grad():
    output = conv_output + input_tensor
    print(output)
