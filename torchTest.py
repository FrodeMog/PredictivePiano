import torch
import torch.nn as nn

# Print found devices
devices = [d for d in range(torch.cuda.device_count())] 
device_names  = [torch.cuda.get_device_name(d) for d in devices]
print(device_names)

# Cuda device to use .to(device)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

""" simple test example
x = torch.empty(1).to(device)
print("empty(1): ", x)

x = torch.empty(3).to(device)
print("empty(3): ", x)

x = torch.empty(2, 3).to(device)

x = torch.randn(2, 3).to(device) # First create on cpu then move to device (slower)
print("rnd x : ", x)
x = torch.randn(2, 3, device=device) # Created directly on device (faster)
print("rnd x : ", x)

print("empty(2, 3): ", x)
print("x size: ", x.size())
print("x Shape: ", x.shape)
"""

""" 
#linear regression example

X = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8 ], dtype=torch.float32, device = device)
Y = torch.tensor([2, 4, 6, 8, 10, 12, 14, 16], dtype=torch.float32, device = device)

W = torch.tensor(0.0, dtype=torch.float32, requires_grad=True, device = device)

#model output
def forward(X):
    return X * W

#loss = MSE
def loss(y, y_pred):
    return ((y_pred - y)**2).mean()

X_test = 5.0

print(f"Prediction before training: f({X_test}) = {forward(X_test).item():.3f}")

#Training
learning_rate = 0.001
n_epochs = 200

for epoch in range(n_epochs):
    #prediction = forward pass
    y_pred = forward(X)

    #loss
    l = loss(Y, y_pred)

    #gradients = backward pass
    l.backward() #dl/dw

    #update weights
    with torch.no_grad():
        W -= learning_rate * W.grad

    #zero gradients
    W.grad.zero_()

    if epoch % 10 == 0:
        print(f"epoch {epoch+1}: w = {W:.3f}, loss = {l:.8f}")

print(f"Prediction after training: f({X_test}) = {forward(X_test).item():.3f}")
"""
""
#Linear regression example with nn.Module
X = torch.tensor([[1], [2], [3], [4], [5], [6], [7], [8 ]], dtype=torch.float32, device = device)
Y = torch.tensor([[2], [4], [6], [8], [10], [12], [14], [16]], dtype=torch.float32, device = device)

n_samples, n_features = X.shape
print(f'n_samples = {n_samples}, n_features = {n_features}')

# 0) create a test sample
X_test = torch.tensor([5], dtype=torch.float32, device = device)

# 1) create a model
class linearRegression(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(linearRegression, self).__init__()
        #define layers
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)
    
input_size, output_size = n_features, n_features
model = linearRegression(input_size, output_size).to(device)
print(f'Prediction before training: f({X_test.item()}) = {model(X_test).item():.3f}')

# 2)) loss and optimizer
learning_rate = 0.01
n_epochs = 100

loss = nn.MSELoss()
optimzier = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3) training loop
for epoch in range(n_epochs):
    #prediction = forward pass
    y_pred = model(X)

    #loss
    l = loss(Y, y_pred)

    #calculate gradients = backward pass
    l.backward() #dl/dw

    #update weights
    optimzier.step()

    #zero gradients
    optimzier.zero_grad()

    if (epoch + 1) % 10 == 0:
        w, b = model.parameters()
        print(f"epoch {epoch+1}: w = {w[0][0].item():.3f}, loss = {l:.8f}")

print(f"Prediction after training: f({X_test.item()}) = {model(X_test).item():.3f}")
