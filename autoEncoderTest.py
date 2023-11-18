import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from midiDataHandler import MidiDataHandler

# Load the data
data = pd.read_csv('test.csv')

# Convert the string representation of lists to actual lists
data['feature'] = data['feature'].apply(eval)
data['label'] = data['label'].apply(eval)

# Scale the data to the range [0, 1]
scaler = MinMaxScaler()

# Convert the features and labels to PyTorch tensors
feature_tensor = torch.tensor(scaler.fit_transform(np.array(data['feature'].tolist())), dtype=torch.float32)
label_tensor = torch.tensor(scaler.transform(np.array(data['label'].tolist())), dtype=torch.float32)

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(5, 5),
            nn.ReLU(),
            nn.Linear(5, 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 5),
            nn.ReLU(),
            nn.Linear(5, 5),
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
for epoch in range(10000):
    # Forward pass
    outputs = model(feature_tensor)
    loss = nn.MSELoss()(outputs, label_tensor)

    # Backward pass and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch {epoch+1}/{10000}, Loss: {loss.item()}')

# Define the input array
input_array = np.array([69, 60, 65, 64, 69])

# Reshape the input to have 5 features
input_array = input_array.reshape(1, -1)

# Scale the input to the range [0, 1] using the same scaler used for training
input_scaled = scaler.transform(input_array)

# Convert the input to a PyTorch tensor
input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

# Pass the input through the model
output = model(input_tensor)

# The output is a tensor. If you want to convert it back to a numpy array, you can do so with the .detach().numpy() methods
output_array = output.detach().numpy()

# Convert the output back to the original space
output_original = scaler.inverse_transform(output_array)

# Clip the values to the MIDI range
output_clipped = np.clip(output_original, 0, 127)

# Round the values to the nearest integer and convert to int
output_rounded = np.round(output_clipped).astype(int)

handler = MidiDataHandler()
print(handler.midi_notes_to_piano(input_array[0]))
print(handler.midi_notes_to_piano(output_rounded[0]))