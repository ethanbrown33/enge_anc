# Use the sounddevice module
# http://python-sounddevice.readthedocs.io/en/0.3.10/

import numpy as np
import sounddevice as sd
import time



def callback_in1(indata, frames, time, status):
    if status:
        print(status, flush=True)
    audio_data['reference'] = indata

def callback_in2(indata, frames, time, status):
    if status:
        print(status, flush=True)
    audio_data['error'] = indata

def callback_out(outdata, frames, time, status):
    if status:
        print(status, flush=True)
    
    if 'reference' in audio_data and 'error' in audio_data:
        # Mix audio data from two input devices
        #process two inputs
        mixed_audio = 0.5 * audio_data['reference'] + 0.5 * audio_data['error']
        outdata[:] = mixed_audio
    else:
        outdata[:] = np.zeros_like(outdata)

# Set the parameters for audio input and output
sample_rate1 = 44100  # You can adjust this value
sample_rate2 = 48000
blocksize = 1024

# Record and play audio for 5 seconds
duration = 5  # Set the desired duration in seconds

# Define audio data dictionary to store input audio from two devices
audio_data = {}

print(sd.query_devices())

with sd.InputStream(callback=callback_in1, blocksize=blocksize, channels=1, samplerate=sample_rate2, device=1), \
     sd.InputStream(callback=callback_in2, blocksize=blocksize, channels=1, samplerate=sample_rate1, device=2), \
     sd.OutputStream(callback=callback_out, blocksize=blocksize, channels=1, samplerate=sample_rate1, device=4):

    print(f"Real-time audio playback from two microphones to one speaker for {duration} seconds.")
    start_time = time.time()
    while time.time() - start_time < duration:
        pass

print(audio_data)

