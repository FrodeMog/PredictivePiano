import mido

class PianoInput:
    _input_device = None
    _last_note = None

    @classmethod
    def get_input_device(cls):
        return cls._input_device

    @classmethod
    def set_input_device(cls, input_device):
        cls._input_device = input_device

    @classmethod
    def get_last_note(cls):
        return cls._last_note
    
    @classmethod
    def set_last_note(cls, note):
        cls._last_note = note

    def __init__(self, input_device=None):
        self.last_note = None
        if input_device is None:
            input_device = PianoInput.get_first_working_port()
        print("Available MIDI input devices:")
        print(mido.get_input_names())
        PianoInput.set_input_device(input_device)

    def get_first_working_port():
        for port_name in mido.get_input_names():
            try:
                with mido.open_input(port_name):
                    return port_name
            except:
                pass
        return None

    async def get_midi_out(self, callback):
        try:
            with mido.open_input(PianoInput.get_input_device()) as port:
                print("Using {}".format(port))
                print('MIDI port opened:', port)
                while True:
                    message = port.receive()
                    if message.type in ['note_on', 'note_off']:
                        callback(message)
                        self.set_last_note(message.note)
        except ValueError:
            print(f"Could not find {PianoInput.get_input_device()}. Available ports are:")
            print(mido.get_input_names())

def note_handler(note: mido.Message) -> int:
    """
    Midi message event handler
    """
    if note.type in ["note_on", "note_off"]:
        return note.note