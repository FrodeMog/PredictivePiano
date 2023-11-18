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
num_files = 3

# Loop through random files
for _ in range(num_files):
    # Set the file path
    file_name = random.choice(os.listdir(midi_folder_path))
    file_path = os.path.join(midi_folder_path, file_name)

    # Call the read_midi_file method and print the result
    result = handler.read_midi_file(file_path)

    print("\nfilename:", file_name)
    #print("midi file:", file_path)
    #print("Tempo:", handler.get_tempo(file_path))
    #print("Time signature:", handler.get_time_signature(file_path))
    #print("0 Index notes: ", handler.midi_notes_to_piano(handler.get_notes_from_index(file_path, 0, 5)))
    #print("Random index notes: ", handler.midi_notes_to_piano(handler.get_notes_from_random(file_path, 5)))
    for _ in range(5):
        pairOfNote = handler.get_pair_of_notes_from_random(file_path, 5)
        print("Random note pairs as midi\n", 
            "first note sequence:\t",pairOfNote[0], 
            "\n second note sequence:\t", pairOfNote[1])
        t = torch.tensor(pairOfNote)
        t_np = t.numpy()
        df = pd.DataFrame(t_np)
        
        # Check if the file exists
        file_exists = os.path.isfile("test.csv")
        
        # Write to the file, appending if it already exists and not writing the header if it already exists
        with open("test.csv", 'a') as f:
            df.to_csv(f, header=not file_exists, index=False)
        #print("Random Pair notes as piano keys: ", handler.midi_notes_to_piano(pairOfNote[0]), handler.midi_notes_to_piano(pairOfNote[1]))
    #print("Dataset pair notes: ", handler.dataset_pair(file_path, 5))
