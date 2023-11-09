import mido

class PianoInput:
    def test_function(self):
        print("Piano input test function called.")
        print("Available MIDI input devices:")
        print(mido.get_input_names())
        # Set up MIDI input
        try:
            with mido.open_input('VirtualMidi Extra 2', callback=note_handler) as port:
                print("Using {}".format(port))
                print('MIDI port opened:', port)
                while True:
                    message = port.receive()
                    if message.type == 'note_on':
                        print('Note on:', message.note)
                    elif message.type == 'note_off':
                        print('Note off:', message.note)
        except ValueError:
            print("Could not find Digital Piano. Available ports are:")
            print(mido.get_input_names())

            print("Available MIDI input devices:")
            print(mido.get_input_names())


def note_handler(note: mido.Message) -> None:
    """
    Midi message event handler
    """
    if note.type in ["note_on", "note_off"]:
        print(note)