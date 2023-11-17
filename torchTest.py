import torch

# Print found devices
devices = [d for d in range(torch.cuda.device_count())] 
device_names  = [torch.cuda.get_device_name(d) for d in devices]
print(device_names)

# Cuda device to use .to(device)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


x = torch.empty(1).to(device)
print("empty(1): ", x)

x = torch.empty(3).to(device)
print("empty(3): ", x)

x = torch.empty(2, 3).to(device)

print("empty(2, 3): ", x)
print("x size: ", x.size())
print("x Shape: ", x.shape)