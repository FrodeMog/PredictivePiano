import asyncio
import mido
from PianoInput import PianoInput

async def main():
    piano_input = PianoInput()
    await piano_input.get_midi_out(callback=handle_message)

def handle_message(note):
    print(f"Received message: {note}")

if __name__ == '__main__':
    asyncio.run(main())
