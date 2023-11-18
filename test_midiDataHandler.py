import os
from midiDataHandler import MidiDataHandler
import random

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
        print("Random Pair notes as midi notes: ", pairOfNote[0], pairOfNote[1])
        print("Random Pair notes as piano keys: ", handler.midi_notes_to_piano(pairOfNote[0]), handler.midi_notes_to_piano(pairOfNote[1]))
    #print("Dataset pair notes: ", handler.dataset_pair(file_path, 5))
