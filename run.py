import gradio as gr
import asyncio
import threading
from PianoInput import PianoInput

piano_input = PianoInput()
notes = []

def handle_message(message):
    if message.type == "note_on":
        notes.append(message.note)
        print("note: " + str(message.note))

def note(input_text):
    if notes:
        return "notes: " + ', '.join(map(str, notes))
    else:
        return "No notes received yet."

iface = gr.Interface(
    fn=note,
    inputs="text",
    outputs=gr.Textbox(max_lines=10, autoscroll=True)
)

def run_midi_input():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(piano_input.get_midi_out(callback=handle_message))

def run_gradio_interface():
    iface.launch()

midi_thread = threading.Thread(target=run_midi_input)
gradio_thread = threading.Thread(target=run_gradio_interface)

midi_thread.start()
gradio_thread.start()