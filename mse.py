import torch
import torch.nn as nn

mse_loss = nn.MSELoss()
output = torch.randn(3, 5, requires_grad=True)
target = torch.randn(3, 5)
loss = mse_loss(output, target)
print(loss)