import torch
import torch.nn as nn


class ActivationFunction(nn.Module):
    def __init__(self):
        super().__init__()

    def softmax_function(self, x):
        x_exp_value = torch.exp(x)
        partition = x_exp_value.sum(0, keepdim=True)
        return x_exp_value / partition

    def stable_softmax_function(self, x):
        c = torch.max(x)
        x_exp_value = torch.exp(x - c)
        partition = x_exp_value.sum(0, keepdim=True)
        return x_exp_value / partition


data = torch.Tensor([1, 2, 3])

softmax = ActivationFunction()
output = softmax.softmax_function(data)
print(output)

output = softmax.stable_softmax_function(data)
print(output)
