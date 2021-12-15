import numpy as np
import pyaudio as pa
import os, time
import nemo
import nemo.collections.asr as nemo_asr
import torch
from queue import Queue
from ruamel.yaml import YAML
from speech2text import *

from threading import Thread
# sample rate, Hz
SAMPLE_RATE = 16000
FRAME_LEN = 4.0
# number of audio channels (expect mono signal)
CHANNELS = 1

CHUNK_SIZE = int(FRAME_LEN*SAMPLE_RATE)
# asr = FrameASR(model_definition = {
#                    'sample_rate': SAMPLE_RATE,
#                    'AudioToMelSpectrogramPreprocessor': cfg.preprocessor,
#                    'JasperEncoder': cfg.encoder,
#                    'labels': cfg.decoder.vocabulary
#                },
#                frame_len=FRAME_LEN, frame_overlap=2, 
#                offset=4)


config = 'config/quartznet12x1_abcfjwz.yaml'
encoder_checkpoint = 'model_vietasr/checkpoints/JasperEncoder-STEP-1312684.pt'
decoder_checkpoint = 'model_vietasr/checkpoints/JasperDecoderForCTC-STEP-1312684.pt'

neural_factory = restore_model(config, encoder_checkpoint, decoder_checkpoint)
print('restore model checkpoint done!')
signals = Queue()

def loop_infer(model, queue_signal):
    while True:
        signal = queue_signal.get()
        if signal is not None:
            greedy_hypotheses, beam_hypotheses = neural_factory.infer_signal(signal)
            print(beam_hypotheses)

if __name__ == "__main__":
    Thread(target = loop_infer, args=(neural_factory, signals)).start()
    p = pa.PyAudio()
    print('Available audio input devices:')
    input_devices = []
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        if dev.get('maxInputChannels'):
            input_devices.append(i)
            print(i, dev.get('name'))

    if len(input_devices):
        dev_idx = -2
        while dev_idx not in input_devices:
            print('Please type input device ID:')
            dev_idx = int(input())

        empty_counter = 0

        def callback(in_data, frame_count, time_info, status):
            global empty_counter
            signal = np.frombuffer(in_data, dtype=np.int16)
            signals.put(signal)
            return (in_data, pa.paContinue)

        stream = p.open(format=pa.paInt16,
                        channels=CHANNELS,
                        rate=SAMPLE_RATE,
                        input=True,
                        input_device_index=dev_idx,
                        stream_callback=callback,
                        frames_per_buffer=CHUNK_SIZE)

        print('Listening...')
        stream.start_stream()

    # Interrupt kernel and then speak for a few more words to exit the pyaudio loop !
        try:
            while stream.is_active():
                time.sleep(0.1)
        finally:        
            stream.stop_stream()
            stream.close()
            p.terminate()

            print()
            print("PyAudio stopped")

    else:
        print('ERROR: No audio input device found.')
