import asyncio
from PianoInput import PianoInput
piano_input = PianoInput()

def handle_message(message):
    note = message.note
    print("note: " + str(note))
    last_note = piano_input.get_last_note()
    if last_note is not None:
        print("last note: " + str(last_note))
    else:
        print("last note: None")
        
async def main():
    await piano_input.get_midi_out(callback=handle_message)

if __name__ == '__main__':
    asyncio.run(main())