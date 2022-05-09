import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from envs.gym_wrapper import *


loss = nn.CrossEntropyLoss()

# # Example of target with class indices
# input = torch.randn(3, 5, requires_grad=True)
# print(input)
# target = torch.empty(3, dtype=torch.long).random_(5)
# print(target)
# output = loss(input, target)
# print(output)
# output.backward()
# print(input.grad)

# Example of target with class probabilities
input = torch.randn(3, 5, requires_grad=True)
print(input)
target = torch.randn(3, 5).softmax(dim=1)
print(target)
output = loss(input, target)
print(output)
output.backward()
print(input.grad)