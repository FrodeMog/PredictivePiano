import mido
import random

#Track 0 piano template
#Track 1 Right hand
#Track 2 Left hand

class MidiDataHandler:
    _midiFolderPATH = None

    def __init__(self, midiFolderPATH=None):
        self.midiFolderPATH = midiFolderPATH

    def read_midi_file(self, file_path):
        """
        Reads a midi file and returns a list of messages
        """
        try:
            mid = mido.MidiFile(file_path)
            return list(mid)
        except Exception as e:
            print(f"Error reading MIDI file {file_path}: {e}")

    def get_tempo(self, file_path):
        """
        Returns the tempo of the MIDI file
        """
        try:
            mid = mido.MidiFile(file_path)
            for msg in mid:
                if msg.type == 'set_tempo':
                    return mido.tempo2bpm(msg.tempo)
        except Exception as e:
            print(f"Error getting tempo from MIDI file {file_path}: {e}")

    #Beats per signature of the numerator/denominator. 4/4 = 4 beats per measure
    def get_time_signature(self, file_path):
        """
        Returns the time signature of the MIDI file
        """
        try:
            mid = mido.MidiFile(file_path)
            for msg in mid:
                if msg.type == 'time_signature':
                    return f"{msg.numerator}/{msg.denominator}"
        except Exception as e:
            print(f"Error getting time signature from MIDI file {file_path}: {e}")

    def get_notes_from_index(self, file_path, start_index, num_notes=5, channel_number=1):
        """
        Returns a list of num_notes notes starting from the specified index in the MIDI file
        The channel_number parameter specifies which channel's notes you want to get.
        Typically, channel 1 is for the right hand and channel 2 is for the left hand.
        """
        try:
            mid = mido.MidiFile(file_path)

            if channel_number >= len(mid.tracks):
                raise ValueError("Invalid channel number")
            
            # construct a note array: [note, velocity, on_tick, duration]
            # note is the midi note number. 0-127 (88 keys on a piano with odd translation)
            # velocity is the volume of the note. 0-127
            # on_tick is the tick when the note was turned on
            # duration is the number of ticks the note was on for
            notes = []
            note_on_dict = {}
            current_tick = 0
            for msg in mid.tracks[channel_number]:
                current_tick += msg.time
                if msg.type == 'note_on':
                    # Store the note and the tick when it was turned on
                    note_on_dict[msg.note] = [msg.note, msg.velocity, current_tick]
                elif msg.type == 'note_off' and msg.note in note_on_dict:
                    # Calculate the duration
                    duration = current_tick - note_on_dict[msg.note][2]
                    # Append the note, velocity, on_tick and duration to the notes list
                    notes.append(note_on_dict[msg.note] + [duration])
                    # Remove the note from the dictionary
                    del note_on_dict[msg.note]


            # Ensure there are enough notes to select from
            if start_index + num_notes > len(notes):
                raise ValueError("Not enough notes in the MIDI file to select from")

            return notes[start_index : start_index + num_notes]

        except Exception as e:
            print(f"Error getting notes from MIDI file {file_path}: {e}")

    def get_notes_from_random(self, file_path, num_notes=5, channel_number=1):
        """
        Returns a list of num_notes notes starting from a random index in the MIDI file.
        The channel_number parameter specifies which channel's notes you want to get.
        Typically, channel 1 is for the right hand and channel 2 is for the left hand.
        """
        try:
            mid = mido.MidiFile(file_path)
            if channel_number >= len(mid.tracks):
                raise ValueError("Invalid channel number")

            # Get the total number of notes in the specified channel
            total_notes = len([msg.note for msg in mid.tracks[channel_number] if msg.type == 'note_on'])

            # Ensure there are enough notes to select from
            if total_notes <= num_notes:
                raise ValueError("Not enough notes in the MIDI file to select from")

            # Select a random start index from the first note to the last possible start note
            start_index = random.randint(0, total_notes - num_notes)

            return self.get_notes_from_index(file_path, start_index, num_notes, channel_number)

        except Exception as e:
            print(f"Error getting notes from MIDI file {file_path}: {e}")
    
    def get_pair_of_notes_from_random(self, file_path, num_notes=5, channel_number=1):
        """
        Returns a pair of arrays of num_notes notes starting from a random index in the MIDI file.
        The channel_number parameter specifies which channel's notes you want to get.
        Typically, channel 1 is for the right hand and channel 2 is for the left hand.
        """
        mid = mido.MidiFile(file_path)
        if len(mid.tracks) != 3:
            raise ValueError("Invalid number of tracks in the MIDI file")

        if channel_number >= len(mid.tracks):
            raise ValueError("Invalid channel number")

        # Get the total number of notes in the specified channel
        total_notes = len([msg.note for msg in mid.tracks[channel_number] if msg.type == 'note_on'])

        # Ensure there are enough notes to select from
        if total_notes <= 2 * num_notes:
            raise ValueError("Not enough notes in the MIDI file to select from")

        # Calculate the maximum possible start index for the pair of arrays
        max_start_index = total_notes - 2 * num_notes

        # Select a random start index within the bounds
        start_index = random.randint(0, max_start_index)

        notes1 = self.get_notes_from_index(file_path, start_index, num_notes, channel_number)
        notes2 = self.get_notes_from_index(file_path, start_index + num_notes, num_notes, channel_number)
        first_note_on_tick = notes1[0][2]
        for note in notes1:
            note[2] -= first_note_on_tick
        for note in notes2:
            note[2] -= first_note_on_tick

        return (notes1, notes2)
    
    def dataset_pair(self, file_path, channel_number, num_notes=5):
        """
        Returns a 2D array representing an 88-key piano, where each value is True if the corresponding key is in the array and False otherwise.
        The channel_number parameter specifies which channel's notes you want to get.
        Typically, channel 1 is for the right hand and channel 2 is for the left hand.
        """
        # Get a pair of arrays of notes from a random index in the MIDI file
        pair_of_notes = self.get_pair_of_notes_from_random(file_path, num_notes, channel_number)

        # Initialize a 2D array representing an 88-key piano with all values set to False
        piano_keys = [[False for _ in range(88)] for _ in range(2)]

        # Set the value to True for each key in the pair of arrays
        for i in range(2):
            for note in pair_of_notes[i]:
                # Ensure the note is within the range of an 88-key piano
                if 21 <= note <= 108:
                    piano_keys[i][note - 21] = True

        return piano_keys

    #Converts a midi note to its corresponding piano key name
    def midi_note_to_piano(self, midi_note):
        """
        Converts a MIDI note number to its corresponding piano key name
        """
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        octave = midi_note // 12 - 1
        note = notes[midi_note % 12]
        return note + str(octave)
    
    #Converts a list of midi notes to their corresponding piano key names
    def midi_notes_to_piano(self, midi_notes):
        """
        Converts a list of MIDI note numbers to their corresponding piano key names
        """
        return [self.midi_note_to_piano(note) for note in midi_notes]
