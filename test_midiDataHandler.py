import os
from midiDataHandler import MidiDataHandler

# Create an instance of the MidiDataHandler class
handler = MidiDataHandler()

# Set the midi folder path
midi_folder_path = "midi_files/pop"

# Set the file path
file_path = os.path.join(midi_folder_path, "ABBA_Honey_Honey.mid")

# Call the read_midi_file method and print the result
result = handler.read_midi_file(file_path)
print(result)