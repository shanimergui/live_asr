import pyaudio
import webrtcvad
from wav2vec2_inference import Wave2Vec2Inference
import numpy as np
import threading
import copy
import time
from sys import exit
import contextvars
from queue import Queue
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
all_text = ''


class LiveWav2Vec2():
    exit_event = threading.Event()

    def __init__(self, model_name, device_name="default"):
        self.model_name = model_name
        self.device_name = device_name

    def stop(self):
        """stop the asr process"""
        try:
            LiveWav2Vec2.exit_event.set()

            self.asr_input_queue.put("close")
            print("asr stopped")
        except:
            pass
    def start(self):
        """start the asr process"""
        self.asr_output_queue = Queue()
        self.asr_input_queue = Queue()
        self.asr_process = threading.Thread(target=LiveWav2Vec2.asr_process, args=(
            self.model_name, self.asr_input_queue, self.asr_output_queue))
        self.asr_process.start()
        time.sleep(5)  # start vad after asr model is loaded
        self.vad_process = threading.Thread(target=LiveWav2Vec2.vad_process, args=(
            self.device_name, self.asr_input_queue,))
        self.vad_process.start()

    def vad_process(device_name, asr_input_queue):
        vad = webrtcvad.Vad()
        vad.set_mode(1)

        audio = pyaudio.PyAudio()
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        # A frame must be either 10, 20, or 30 ms in duration for webrtcvad
        FRAME_DURATION = 30
        CHUNK = int(RATE * FRAME_DURATION / 1000)
        RECORD_SECONDS = 50

        microphones = LiveWav2Vec2.list_microphones(audio)
        selected_input_device_id = LiveWav2Vec2.get_input_device_id(
            device_name, microphones)

        stream = audio.open(input_device_index=selected_input_device_id,
                            format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            frames_per_buffer=CHUNK)

        frames = b''
        while True:
            if LiveWav2Vec2.exit_event.is_set():
                break
            frame = stream.read(CHUNK, exception_on_overflow=False)
            is_speech = vad.is_speech(frame, RATE)
            if is_speech:
                frames += frame
            else:
                if len(frames) > 1:
                    asr_input_queue.put(frames)
                frames = b''
        stream.stop_stream()
        stream.close()
        audio.terminate()

    def file_to_text(self,file_name):
        wave2vec_asr = Wave2Vec2Inference(self.model_name, use_lm_if_possible=True)
        return wave2vec_asr.file_to_text(file_name)

    def asr_process(model_name, in_queue, output_queue,filename=None):
        wave2vec_asr = Wave2Vec2Inference(model_name, use_lm_if_possible=True)

        print("\nlistening to your voice\n")
        while True:
            audio_frames = in_queue.get()
            if audio_frames == "close":
                break


            float64_buffer = np.frombuffer(
                audio_frames, dtype=np.int16) / 32767
            start = time.perf_counter()

            text, confidence = wave2vec_asr.buffer_to_text(float64_buffer)
            text = text.lower()
            inference_time = time.perf_counter() - start
            sample_length = len(float64_buffer) / 16000  # length in sec
            if text != "":
                output_queue.put([text, sample_length, inference_time, confidence])

    def get_input_device_id(device_name, microphones):
        for device in microphones:
            if device_name in device[1]:
                return device[0]

    def list_microphones(pyaudio_instance):
        info = pyaudio_instance.get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount')

        result = []
        for i in range(0, numdevices):
            if (pyaudio_instance.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                name = pyaudio_instance.get_device_info_by_host_api_device_index(
                    0, i).get('name')
                result += [[i, name]]
        return result

    def get_last_text(self):
        """returns the text, sample length and inference time in seconds."""
        return self.asr_output_queue.get()



asr = LiveWav2Vec2('jonatasgrosman/wav2vec2-large-xlsr-53-english')
window = tk.Tk()
table = ttk.Treeview(columns=('sample_length', 'inference_time', 'confidence', 'text'), show="headings")


def end():
    asr.stop()
    with open('output.txt', mode ='w',encoding='utf-8') as f:
        f.write("")
    with open('output.txt', mode ='w',encoding='utf-8') as f:
        f.write(all_text)
    f.close()
    for i in table.get_children():
        table.delete(i)
    print('Listening stopped successfully')


def get_recording():
    file_path = filedialog.askopenfilename()
    if file_path:
        text, sample_length, inference_time, confidence = asr.file_to_text(file_path)
        table.insert("", "end", values=(sample_length, round(inference_time, 2), round(confidence.item(), 2), text))
        table.pack()
        window.update()

def live():
    global all_text
    asr.start()
    while True:
        text, sample_length, inference_time, confidence = asr.get_last_text()
        all_text = all_text + ' ' + text
        table.insert("", "end", values=(sample_length, round(inference_time, 2), round(confidence.item(), 2), text))
        table.pack()
        window.update()



if __name__ == "__main__":

    window.title('Live ASR')
    buttons = tk.Frame(window)

    b1 = tk.Button(buttons, text='Stop & Save', command=end)
    b1.pack(side='left')
    b2 = tk.Button(buttons, text='Upload audio', command=get_recording)
    b2.pack(side='left')
    b3 = tk.Button(buttons, text='Listen Live', command=live)
    b3.pack(side='left')

    scroll = tk.Scrollbar()
    scroll.pack(side=tk.RIGHT)
    table.heading("#1", text='sample_length')
    table.heading("#2", text='inference_time')
    table.heading("#3", text='confidence')
    table.heading("#4", text='text')
    header = tk.Label(text='Live ASR', font=('gisha', 20, 'bold'))
    header.pack()
    table.pack()
    buttons.pack()
    window.mainloop()
