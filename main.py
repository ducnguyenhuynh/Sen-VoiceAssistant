"""the interface to interact with wakeword model"""
import pyaudio
import threading
import time
import argparse
import wave
import torchaudio
import torch
import numpy as np
from neuralnet.dataset import get_featurizer
from threading import Event

import os
import subprocess
import random
from os.path import join, realpath
import pyaudio as pa

import nemo
import nemo.collections.asr as nemo_asr
from queue import Queue
from ruamel.yaml import YAML
from speech2text import *
from threading import Thread
from SenBot.main import Assistant
bot = Assistant()
    
class Listener:
    # global Flag_wakeup
    def __init__(self, sample_rate=8000, record_seconds=2):
        self.chunk = 1024
        self.sample_rate = sample_rate
        self.record_seconds = record_seconds
        self.p = pyaudio.PyAudio()
        self.Flag_wakeup = 0
        self.stream = self.p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=self.sample_rate,
                        input=True,
                        input_device_index=-2,
                        frames_per_buffer=self.chunk)

    def listen(self, queue):
        while not self.Flag_wakeup:
            data = self.stream.read(self.chunk, exception_on_overflow=False)
            queue.append(data)


    def run(self, queue):
        thread = threading.Thread(target=self.listen, args=(queue,), daemon=True)
        thread.start()
        print("\nWake Word Engine is now listening... \n")


class WakeWordEngine:
    
    # global Flag_wakeup
    def __init__(self, model_file):
        self.listener = Listener(sample_rate=8000, record_seconds=2)

        self.model = torch.jit.load(model_file)
        self.model.eval().to('cpu')  #run on cpu
        self.featurizer = get_featurizer(sample_rate=8000)
        self.audio_q = list()
        self.Flag_wakeup = 0


    def save(self, waveforms, fname="wakeword_temp"):
        wf = wave.open(fname, "wb")
        # set the channels
        wf.setnchannels(1)
        # set the sample format
        wf.setsampwidth(self.listener.p.get_sample_size(pyaudio.paInt16))
        # set the sample rate
        wf.setframerate(8000)
        # write the frames as bytes
        wf.writeframes(b"".join(waveforms))
        # close the file
        wf.close()
        return fname


    def predict(self, audio):
        with torch.no_grad():
            fname = self.save(audio)
            waveform, _ = torchaudio.load(fname)  # don't normalize on train
            mfcc = self.featurizer(waveform).transpose(1, 2).transpose(0, 1)

            # TODO: read from buffer instead of saving and loading file
            # waveform = torch.Tensor([np.frombuffer(a, dtype=np.int16) for a in audio]).flatten()
            # mfcc = self.featurizer(waveform).transpose(0, 1).unsqueeze(1)

            out = self.model(mfcc)
            pred = torch.round(torch.sigmoid(out))
            return pred.item()

    def inference_loop(self, action):

        while  not self.Flag_wakeup:
            if len(self.audio_q) > 15:  # remove part of stream
                diff = len(self.audio_q) - 15
                for _ in range(diff):
                    self.audio_q.pop(0)
                output =  self.predict(self.audio_q)
                # output = 1
                if output == 1:
                    self.Flag_wakeup = 1
                    self.listener.Flag_wakeup = 1
                action(output)
                
            elif len(self.audio_q) == 15:
                output =  self.predict(self.audio_q)
                # output = 1
                if output == 1:
                    self.Flag_wakeup = 1
                    self.listener.Flag_wakeup = 1
                action(output)
            time.sleep(0.01)

    def run(self, action):
        
        self.listener.run(self.audio_q)

        thread = threading.Thread(target=self.inference_loop,
                                    args=(action,), daemon=True)
        thread.start()
        
class DemoAction:
    """This demo action will just randomly say Arnold Schwarzenegger quotes
        args: sensitivty. the lower the number the more sensitive the
        wakeword is to activation.
    """
    # global Flag_wakeup
    def __init__(self, sensitivity=10):
        # import stuff here to prevent engine.py from 
        # importing unecessary modules during production usage
        

        # sample rate, Hz
        SAMPLE_RATE = 16000
        FRAME_LEN = 3.5
        # number of audio channels (expect mono signal)
        CHANNELS = 1
        CHUNK_SIZE = int(FRAME_LEN*SAMPLE_RATE)
        
        # self.random = random
        # self.subprocess = subprocess
        # self.detect_in_row = 0

        # self.sensitivity = sensitivity

        config = 'config/quartznet12x1_abcfjwz.yaml'
        encoder_checkpoint = 'model_vietasr/checkpoints/JasperEncoder-STEP-1312684.pt'
        decoder_checkpoint = 'model_vietasr/checkpoints/JasperDecoderForCTC-STEP-1312684.pt'
        self.neural_factory = restore_model(config, encoder_checkpoint, decoder_checkpoint)
        print('restore model checkpoint done!')
        self.signals = Queue()
        p = pa.PyAudio()
        empty_counter = 0
            
        def callback(in_data, frame_count, time_info, status):
                global empty_counter
                signal = np.frombuffer(in_data, dtype=np.int16)
                self.signals.put(signal)
                return (in_data, pa.paContinue)

        self.stream = p.open(format=pa.paInt16,
                        channels=CHANNELS,
                        rate=SAMPLE_RATE,
                        input=True,
                        input_device_index=-2,
                        stream_callback=callback,
                        frames_per_buffer=CHUNK_SIZE)
        
    def loop_infer(self):
        while True:
            signal = self.signals.get()
            if signal is not None:
                greedy_hypotheses, beam_hypotheses = self.neural_factory.infer_signal(signal)
                # beam_hypotheses
                print(beam_hypotheses)
                bot.do_action(beam_hypotheses)
                beam_hypotheses = "abc"
                # print(beam_hypotheses)

    def __call__(self, prediction):
        if prediction == 1:
            print("Tao day roi")
            Thread(target = self.loop_infer).start()
            print('Listening...')
            self.stream.start_stream()
            
        else:
            print("Khò Khò!")
    def play(self):
        filename = self.random.choice(self.arnold_mp3)
        try:
            print("playing", filename)
            self.subprocess.check_output(['play', '-v', '.1', filename])
        except Exception as e:
            print(str(e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="demoing the wakeword engine")
    parser.add_argument('--model_file', type=str, default="./optimized_mode.pt",
                        help='optimized file to load. use optimize_graph.py')
    parser.add_argument('--sensitivty', type=int, default=10, required=False,
                        help='lower value is more sensitive to activations')

    args = parser.parse_args()
    wakeword_engine = WakeWordEngine(args.model_file)
    action = DemoAction(sensitivity=10)

    print("""\n*** Make sure you have sox installed on your system for the demo to work!!!
    If you don't want to use sox, change the play function in the DemoAction class
    in engine.py module to something that works with your system.\n
    """)
    # # action = lambda x: print(x)
    wakeword_engine.run(action)
    threading.Event().wait()  

    # time.sleep(1000)
    # threading.Event().set()