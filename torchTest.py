import torch

devices = [d for d in range(torch.cuda.device_count())] 
device_names  = [torch.cuda.get_device_name(d) for d in devices]
print(device_names)


x = torch.empty(1)
print("empty(1): ", x)

x = torch.empty(3)
print("empty(3): ", x)

x = torch.empty(2, 3)
print("empty(2, 3): ", x)
print("x Size: ", x.size())
print("x Shape: ", x.shape)