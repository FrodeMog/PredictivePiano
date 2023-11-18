import mido

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