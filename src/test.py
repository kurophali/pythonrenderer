import torch

x = torch.full((3,2,4),1)
y = torch.full((4),2)
print(torch.dot(x,y))
