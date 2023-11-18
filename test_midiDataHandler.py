import os
from midiDataHandler import MidiDataHandler

# Create an instance of the MidiDataHandler class
handler = MidiDataHandler()

# Set the midi folder path
midi_folder_path = "midi_files/piano"

# Set the file path
file_path = os.path.join(midi_folder_path, "fairy.mid")

# Call the read_midi_file method and print the result
result = handler.read_midi_file(file_path)

print("midi file:", file_path)
print("Tempo:", handler.get_tempo(file_path))
print("Time signature:", handler.get_time_signature(file_path))
print("0 Index notes: ", handler.midi_notes_to_piano(handler.get_notes_from_index(file_path, 0, 5)))
print("Random index notes: ", handler.midi_notes_to_piano(handler.get_notes_from_random(file_path, 5)))
print("Random Pair notes: ", handler.get_pair_of_notes_from_random(file_path, 5))
print("Dataset pair notes: ", handler.dataset_pair(file_path, 5))
