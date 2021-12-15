import numpy as np
import pyaudio as pa
import os, time
import nemo
import nemo.collections.asr as nemo_asr
import torch
from ruamel.yaml import YAML

# sample rate, Hz
SAMPLE_RATE = 16000

# class for streaming frame-based ASR
# 1) use reset() method to reset FrameASR's state
# 2) call transcribe(frame) to do ASR on
#    contiguous signal's frames
class FrameASR:
    
    def __init__(self, model_definition,
                 frame_len=2, frame_overlap=2.5, 
                 offset=10):
        '''
        Args:
          frame_len: frame's duration, seconds
          frame_overlap: duration of overlaps before and after current frame, seconds
          offset: number of symbols to drop for smooth streaming
        '''
        self.vocab = list(model_definition['labels'])
        self.vocab.append('_')
        
        self.sr = model_definition['sample_rate']
        self.frame_len = frame_len
        self.n_frame_len = int(frame_len * self.sr)
        self.frame_overlap = frame_overlap
        self.n_frame_overlap = int(frame_overlap * self.sr)
        timestep_duration = model_definition['AudioToMelSpectrogramPreprocessor']['window_stride']
        for block in model_definition['JasperEncoder']['jasper']:
            timestep_duration *= block['stride'][0] ** block['repeat']
        self.n_timesteps_overlap = int(frame_overlap / timestep_duration) - 2
        self.buffer = np.zeros(shape=2*self.n_frame_overlap + self.n_frame_len,
                               dtype=np.float32)
        self.offset = offset
        self.reset()
        
    def _decode(self, frame, offset=0):
        assert len(frame)==self.n_frame_len
        self.buffer[:-self.n_frame_len] = self.buffer[self.n_frame_len:]
        self.buffer[-self.n_frame_len:] = frame
        logits = infer_signal(asr_model, self.buffer).cpu().numpy()[0]
        # print(logits.shape)
        decoded = self._greedy_decoder(
            logits[self.n_timesteps_overlap:-self.n_timesteps_overlap], 
            self.vocab
        )
        return decoded[:len(decoded)-offset]
    
    @torch.no_grad()
    def transcribe(self, frame=None, merge=True):
        if frame is None:
            frame = np.zeros(shape=self.n_frame_len, dtype=np.float32)
        if len(frame) < self.n_frame_len:
            frame = np.pad(frame, [0, self.n_frame_len - len(frame)], 'constant')
        unmerged = self._decode(frame, self.offset)
        if not merge:
            return unmerged
        return self.greedy_merge(unmerged)
    
    def reset(self):
        '''
        Reset frame_history and decoder's state
        '''
        self.buffer=np.zeros(shape=self.buffer.shape, dtype=np.float32)
        self.prev_char = ''

    @staticmethod
    def _greedy_decoder(logits, vocab):
        s = ''
        for i in range(logits.shape[0]):
            s += vocab[np.argmax(logits[i])]
        return s

    def greedy_merge(self, s):
        s_merged = ''
        
        for i in range(len(s)):
            if s[i] != self.prev_char:
                self.prev_char = s[i]
                if self.prev_char != '_':
                    s_merged += self.prev_char
        return s_merged

# duration of signal frame, seconds
# FRAME_LEN = 1.0
# number of audio channels (expect mono signal)
# CHANNELS = 1

# CHUNK_SIZE = int(FRAME_LEN*SAMPLE_RATE)
# asr = FrameASR(model_definition = {
#                    'sample_rate': SAMPLE_RATE,
#                    'AudioToMelSpectrogramPreprocessor': cfg.preprocessor,
#                    'JasperEncoder': cfg.encoder,
#                    'labels': cfg.decoder.vocabulary
#                },
#                frame_len=FRAME_LEN, frame_overlap=2, 
#                offset=4)

if __name__ == "__main__":

    p = pa.PyAudio()
    print('Available audio input devices:')
    input_devices = []
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        if dev.get('maxInputChannels'):
            input_devices.append(i)
            print(i, dev.get('name'))
    
    print(len(input_devices))
    if len(input_devices):
        dev_idx = -2
        while dev_idx not in input_devices:
            print('Please type input device ID:')
            dev_idx = int(input())

        empty_counter = 0

        def callback(in_data, frame_count, time_info, status):
            global empty_counter
            signal = np.frombuffer(in_data, dtype=np.int16)
            text = asr.transcribe(signal)
            if len(text):
                print(text,end='')
                empty_counter = asr.offset
            elif empty_counter > 0:
                empty_counter -= 1
                if empty_counter == 0:
                    print(' ',end='')
            return (in_data, pa.paContinue)

        stream = p.open(format=pa.paInt16,
                        channels=CHANNELS,
                        rate=SAMPLE_RATE,
                        input=True,
                        input_device_index=dev_idx,
                        stream_callback=callback,
                        frames_per_buffer=CHUNK_SIZE)

        print('Listening...')
