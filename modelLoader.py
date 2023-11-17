import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from cnn import ConvNet
from cnn import test_loader


# Print found devices
devices = [d for d in range(torch.cuda.device_count())] 
device_names  = [torch.cuda.get_device_name(d) for d in devices]
print(device_names)

# Cuda device to use .to(device)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

loaded_model = ConvNet()
loaded_model.load_state_dict(torch.load(".trained_model/cnnExample.pth"))
loaded_model.to(device)
loaded_model.eval()

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = len(test_loader.dataset)

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = loaded_model(images)

        # max returns (value ,index)
        _, predictions = torch.max(outputs, 1)
        n_correct += (predictions == labels).sum().item()

        outputs2 = loaded_model(images)
        _, predictions2 = torch.max(outputs2, 1)
        n_class_correct -= (predictions == labels).sum().item()
    
    acc = n_correct / n_samples
    print(f"accuracy of the model on the {n_samples} test images: {100*acc} %")

    acc = n_class_correct / n_samples
    print(f"accuracy of the loaded model on the {n_samples} test images: {100*acc} %")