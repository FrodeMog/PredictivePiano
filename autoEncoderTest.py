import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch import nn


# Load the data
data = pd.read_csv('test.csv')

# Convert the string representation of lists to actual lists
data['feature'] = data['feature'].apply(eval)
data['label'] = data['label'].apply(eval)

# Flatten the data
data_flat = [item for sublist in data['feature'].tolist() for item in sublist]
data_flat += [item for sublist in data['label'].tolist() for item in sublist]

# Scale the data to the range [0, 1]
scaler = MinMaxScaler()
data_flat = scaler.fit_transform(np.array(data_flat).reshape(-1, 1))

# Convert the data to PyTorch tensors
data_tensor = torch.tensor(data_flat, dtype=torch.float32)

# Reshape the data to match the input shape of the model
data_tensor = data_tensor.view(-1, 10)

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 5),
            nn.ReLU(),
            nn.Linear(5, 10),
            nn.Sigmoid()  # because we scaled our data to be in the range [0, 1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
from torch.optim import Adam

# Initialize the autoencoder and the optimizer
model = Autoencoder()
optimizer = Adam(model.parameters(), lr=0.0001)

# Train the autoencoder
for epoch in range(1000):
    # Forward pass
    outputs = model(data_tensor)
    loss = nn.MSELoss()(outputs, data_tensor)

    # Backward pass and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch {epoch+1}/{1000}, Loss: {loss.item()}')

# Define the input array
input_array = np.array([69, 60, 65, 64, 69])

# Pad the input array with zeros until it has 10 elements
input_array = np.pad(input_array, (0, 10 - len(input_array)), 'constant')

# Scale the input to the range [0, 1] using the same scaler used for training
input_scaled = scaler.transform(input_array.reshape(-1, 1))

# Convert the input to a PyTorch tensor
input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

# Reshape the input to match the input shape of the model
input_tensor = input_tensor.view(-1, 10)

# Pass the input through the model
output = model(input_tensor)

# The output is a tensor. If you want to convert it back to a numpy array, you can do so with the .detach().numpy() methods
output_array = output.detach().numpy()

# Convert the output back to the original space
output_original = scaler.inverse_transform(output_array)

# Clip the values to the MIDI range
output_clipped = np.clip(output_original, 0, 127)

# Round the values to the nearest integer
output_rounded = np.round(output_clipped)

print(output_rounded)