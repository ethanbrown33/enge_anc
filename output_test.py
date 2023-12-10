import numpy as np
import sounddevice as sd
import time

start_idx = 0
start_time = time.time()
try:
    samplerate = sd.query_devices(6, 'output')['default_samplerate']

    def callback(outdata, frames, time, status):
        if status:
            print(status)
        global start_idx
        t = (start_idx + np.arange(frames)) / 44100
        t = t.reshape(-1, 1) + 0.01
        outdata[:] = 0.05 * np.sin(2 * np.pi * 220 * t)
        start_idx += frames

    with sd.OutputStream(device=5, channels=1, callback=callback,
                         samplerate=44100):
        while True:
            time.sleep(0)
            if (time.time() - start_time >= 30):
                break
except KeyboardInterrupt:
    print("Keyboard Interrupt")
except Exception as e:
    print("Exception")

print(time.time() - start_time)
print("Program END")
