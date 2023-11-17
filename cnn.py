import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os


# Print found devices
devices = [d for d in range(torch.cuda.device_count())] 
device_names  = [torch.cuda.get_device_name(d) for d in devices]
print(device_names)

# Cuda device to use .to(device)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

#hyperparameters
num_epochs = 10
batch_size = 32
learning_rate = 0.001

#dataset has pilimage images of rtange [0,1]
#need to transform to tensor and normalize to [-1,1]
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#cifar10: 60000 32x32 color images in 10 classes, with 6000 images per class
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(img):
    #unnormalize
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()

#on batch of random training images
dataiter = iter(train_loader)
images, labels = next(dataiter)
img_grid = torchvision.utils.make_grid(images[0:25], nrow=5)
#imshow(img_grid)

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64,64, 3)
        self.fc1 = nn.Linear(64*4*4, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        #N, 3, 32, 32
        x = F.relu(self.conv1(x))   #N, 32, 30, 30
        #print(x.shape)
        x = self.pool(x)            #N, 32, 15, 15  
        #print(x.shape)
        x = F.relu(self.conv2(x))   #N, 64, 13, 13
        #print(x.shape)
        x = self.pool(x)            #N, 64, 6, 6
        #print(x.shape)
        x = F.relu(self.conv3(x))   #N, 64, 4, 4
        #print(x.shape)
        x = torch.flatten(x, 1)     #N, 1024
        #print(x.shape)
        x = F.relu(self.fc1(x))     #N, 64
        #print(x.shape)
        x = self.fc2(x)             #N, 10
        #print(x.shape)
        return x
    
model = ConvNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        #forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        #backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'epoch [{epoch+1}], loss = {running_loss/n_total_steps:.3f}')
print('Finished Training')

PATH = '.trained_model/cnnExample.pth'

# Create the directory if it doesn't exist
directory = os.path.dirname(PATH)
if not os.path.exists(directory):
    os.makedirs(directory)

# Now you can save the model
print('Saving model to ' + PATH)
torch.save(model.state_dict(), PATH)