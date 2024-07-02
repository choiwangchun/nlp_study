import torch
import torch.nn as nn

ce_loss = nn.CrossEntropyLoss()
output = torch.randn(3, 5, requires_grad=True)
target = torch.tensor([1, 0, 3], dtype=torch.int64)
loss = ce_loss(output, target)
print(output)
print(target)
print(loss)