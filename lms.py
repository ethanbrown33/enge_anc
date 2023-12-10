import numpy as np
import sounddevice as sd
import time
import matplotlib.pyplot as plt

sourceData = np.zeros((1136, 1))
w = np.zeros((1136, 1))
DURATION = 10

start_idx = 0

def update_weights(src, err, w, lr):
    error = error_estimation(src, err, w)
    return np.add(w, lr * np.multiply(error, src))

def filter_output(src, w):
    return np.multiply(w, src)

def error_estimation(src, err, w):
    y = filter_output(src, w)
    return np.subtract(err, y)

def callback(indata, outdata, frames, time, status):
    global w
    if status:
        print(status)
    #error = np.max(indata) - np.max(sourceData)
    #adjust = 1 * np.roll(sourceData, round(error))

    w = update_weights(sourceData, indata, w, 0.1)
    outdata[:] = filter_output(sourceData, w)

    #outdata[:] = indata * -0.8

    #outdata[:] = np.roll(sourceData * -5, 500)

def in_callback(indata, frames, time, status):
    global sourceData
    if status:
        print(status)
    sourceData[:] = indata * 2

start_idx = 0
start_time = time.time()
try:
    with sd.Stream(device=(2, 5), channels=1, callback=callback), \
         sd.InputStream(device=1, channels=1, callback=in_callback):
        while True:
            time.sleep(0.1)
            if (time.time() - start_time >= DURATION):
                break
except KeyboardInterrupt:
    print("Keyboard Interrupt")
except Exception as e:
    print("Exception")

print(time.time() - start_time)
print("Program END")

print(sourceData)
