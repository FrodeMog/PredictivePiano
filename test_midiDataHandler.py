import os
from midiDataHandler import MidiDataHandler
import random
import torch as torch
import numpy as np
import pandas as pd

# Create an instance of the MidiDataHandler class
handler = MidiDataHandler()

# Set the midi folder path
midi_folder_path = "midi_files/piano"


# Set the number of random files to process
num_files = 220
num_sequences_per_file = 10

# Initialize an empty DataFrame
data = pd.DataFrame()

for _ in range(num_files):
    # Set the file path
    file_name = random.choice(os.listdir(midi_folder_path))
    file_path = os.path.join(midi_folder_path, file_name)

    # Call the read_midi_file method and print the result
    result = handler.read_midi_file(file_path)

    print("\nfilename:", file_name)

    for _ in range(num_sequences_per_file):
        try:
            pairOfNote = handler.get_pair_of_notes_from_random(file_path, 5, 1)
        except ValueError as e:
            print("Error:", e)
            break
        print("Random note pairs as midi\n", 
            "first note sequence:\t",pairOfNote[0], 
            "\n second note sequence:\t", pairOfNote[1])
        
        # Convert the pair of notes to a DataFrame
        df = pd.DataFrame({
            'feature': [pairOfNote[0]],
            'label': [pairOfNote[1]]
        })
        
        # Append the DataFrame to the data
        data = pd.concat([data, df], ignore_index=True)

# Save the data to a CSV file
data.to_csv("test.csv", index=False)

#print("midi file:", file_path)
#print("Tempo:", handler.get_tempo(file_path))
#print("Time signature:", handler.get_time_signature(file_path))
#print("0 Index notes: ", handler.midi_notes_to_piano(handler.get_notes_from_index(file_path, 0, 5)))
#print("Random index notes: ", handler.midi_notes_to_piano(handler.get_notes_from_random(file_path, 5)))
# Create an empty DataFrame to store the features and labels
#print("Random Pair notes as piano keys: ", handler.midi_notes_to_piano(pairOfNote[0]), handler.midi_notes_to_piano(pairOfNote[1]))
#print("Dataset pair notes: ", handler.dataset_pair(file_path, 5))
