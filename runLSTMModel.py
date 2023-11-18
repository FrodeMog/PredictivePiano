import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from LSTM import LSTM 
import numpy as np
import torch
import mido

# Define the device
device = "cpu"

# Initialize the model with the same parameters
model = LSTM(input_size=1, hidden_size=127, output_size=5)

# Load the state dict previously saved
model.load_state_dict(torch.load('LSTMTest.pth'))

# Move the model to the device
model = model.to(device)

# Make sure to call model.eval() to set dropout and batch normalization layers to evaluation mode before running inference
model.eval()

# Define the input array
input_data = np.array([69, 60, 65, 64, 69])

# Initialize an empty list to store the outputs
outputs = []
outputs = input_data.copy()

# Process the input and append the output to the list
for _ in range(15):
    print("input: ", input_data)
    
    # Convert the input to a PyTorch tensor and reshape it
    input_tensor = torch.tensor(input_data, dtype=torch.float32).view(1, len(input_data), 1)

    # Move the input to the same device as the model
    input_tensor = input_tensor.to(device)

    # Use the model to make a prediction
    with torch.no_grad():
        output = model(input_tensor)

    # Convert the output to a numpy array and inverse transform
    output_np = output.cpu().numpy().flatten()
    print("output ", output_np)

    # Append the output to the outputs list
    outputs = np.concatenate((outputs, output_np))

    # Use the output as the next input
    input_data = output_np

# Create a new MIDI file
mid = mido.MidiFile()

# Create a new MIDI track
track = mido.MidiTrack()

# Add the track to the MIDI file
mid.tracks.append(track)

# For each note in the final output
for note in outputs:
    # Ensure note is an integer
    note = int(note)

    # Add a note_on message to the track
    track.append(mido.Message('note_on', note=note, velocity=64, time=120))

    # Add a note_off message to the track
    track.append(mido.Message('note_off', note=note, velocity=64, time=200))

# Save the MIDI file
mid.save('output.mid')