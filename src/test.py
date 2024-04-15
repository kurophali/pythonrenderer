import torch

d0 = 2
d1 = 3
d2 = 4

# Create an original tensor
original_tensor = torch.rand((d0, d1, d2))

# Create indices tensor
indices = torch.randint(0, d1, size=(d0, 1))

one_hot = torch.nn.functional.one_hot(indices, num_classes=d1).reshape((d0, d1,1)).expand(-1,-1,d2)
print(original_tensor)
print(indices)
print(torch.sum(one_hot * original_tensor, dim=1))
# new_tensor will have shape (d0, d2), where each row corresponds to the elements selected from the original tensor using the indices