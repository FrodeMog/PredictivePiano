import asyncio
from PianoInput import PianoInput

async def main():
    piano_input = PianoInput()
    print(await piano_input.get_midi_out())

if __name__ == '__main__':
    asyncio.run(main())
