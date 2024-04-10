import torch

a = torch.Tensor([[[1,1]]])
b = torch.Tensor([[[0,1]]])
matrices = torch.cat((a,b),1)
print(matrices.numpy())
