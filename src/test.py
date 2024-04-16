import torch

h = 2
w = 2
t = 3
c = 2

attributes = torch.rand((h, w, t, c))
dist = torch.rand(h, w, t)
# inside = torch.randint(0, 2, size=(h, w, t))
inside = torch.nn.functional.one_hot(torch.argmin(dist, dim=2), num_classes=t) # (h, w, t)

closest_orders = torch.argsort(dist, dim=2) + 1 # 0 just means nothing is added? in (h, w, t)
inside_orders = closest_orders * inside # (h,w,t) only the orders that are inside kept its orders. others are 0
selected_ids = inside_orders.argmin(dim=2) # 0 if nothing is inside. the rest are idx+1. in (h,w)
# masks = torch.nn.functional.one_hot(selected_ids, num_classes=t+1)[:,:,1:] # first index that meant nothing are got rid of so it's left as 0 (h,w,t)
masks = torch.nn.functional.one_hot(selected_ids, num_classes=t)
masks = masks[:,:,:,None].expand((-1,-1,-1,c))
selected_attributes = masks * attributes
selected_attributes = selected_attributes.sum(2)
# indices = torch.randint(0, t, size=(h, w, 1))
# one_hot = torch.nn.functional.one_hot(indices, num_classes=t)
# one_hot = one_hot.reshape((h,w,t,1)).expand(-1,-1,-1,c)
print('=============== attributes ===============')
print(attributes)
print('=============== inside ===============')
print(inside)
print('=============== dist ===============')
print(dist)
print('=============== selected_ids ===============')
print(selected_ids)
print('=============== inside_orders ===============')
print(inside_orders)
print('=============== masks ===============')
print(masks)
print('=============== selected_attributes ===============')
print(selected_attributes)


