import torch
import numpy as np
from LSTM import LSTM 
import numpy as np
import torch
import mido

# Define the device
device = "cpu"

# Initialize the model with the same parameters
model = LSTM(input_size=4, hidden_size=127, output_size=4, num_layers=2)

# Load the state dict previously saved
model.load_state_dict(torch.load('LSTMTest2.pth'))

# Move the model to the device
model = model.to(device)

# Make sure to call model.eval() to set dropout and batch normalization layers to evaluation mode before running inference
model.eval()

# Define the input array
input_data = np.array([[79, 100, 0, 144], [78, 100, 120, 144], [76, 100, 240, 239], [76, 100, 480, 239], [76, 100, 720, 264]])

# Initialize an empty list to store the outputs
outputs = []
outputs = input_data.copy()

# Process the input and append the output to the list
for _ in range(20):
    print("input: ", input_data)
    
    # Convert the input to a PyTorch tensor and reshape it
    input_tensor = torch.tensor(input_data, dtype=torch.float32).view(1, len(input_data), 4)

    # Move the input to the same device as the model
    input_tensor = input_tensor.to(device)

    # Use the model to make a prediction
    with torch.no_grad():
        output = model(input_tensor)

    # Convert the output to a numpy array and reshape
    output_np = output.cpu().numpy().flatten().reshape(-1, 4)
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

# Define a scale factor
scale_factor = 0.05

# For each note in the final output
for note_data in outputs:
    # Extract the note, velocity, on_tick, and duration
    note, velocity, on_tick, duration = note_data

    # Scale down on_tick and duration
    on_tick = int(on_tick * scale_factor)
    duration = int(duration * scale_factor)

    # Add a note_on message to the track
    track.append(mido.Message('note_on', note=int(note), velocity=int(velocity), time=on_tick))

    # Add a note_off message to the track
    track.append(mido.Message('note_off', note=int(note), velocity=int(velocity), time=on_tick + duration))

# Save the MIDI file
mid.save('output.mid')