# Import packages
import numpy as np
import sounddevice as sd
import time
import matplotlib.pyplot as plt

blocksize = 1024
weights = np.zeros(blocksize)

# Audio data storage
audio_data = {}

# Initialize plot
plotsize = 4096

x = np.arange(0,plotsize)
src_plot = np.zeros(plotsize)
err_plot = np.zeros(plotsize)
error_plot = np.zeros(plotsize)

fig, ax = plt.subplots()
src_ln, = ax.plot(x, src_plot, animated=True)
err_ln, = ax.plot(x, err_plot, animated=True)
error_ln, = ax.plot(x, error_plot, animated=True)

plt.ylim(-0.3, 0.3)
plt.xlim(0, plotsize)
plt.show(block=False)
plt.pause(0.1)

# Blit plot
bg = fig.canvas.copy_from_bbox(fig.bbox)
ax.draw_artist(src_ln)
ax.draw_artist(err_ln)
ax.draw_artist(error_ln)
fig.canvas.blit(fig.bbox)

# Initialize stream and loop
duration = 10
start_time = time.time()


# Functions
def src_callback(indata, frames, time, status):
    if status:
        print(status, flush=True)
    audio_data['source'] = indata.flatten()

def err_callback(indata, frames, time, status):
    if status:
        print(status, flush=True)
    audio_data['error'] = indata.flatten()

def out_callback(outdata, frames, time, status):
    global weights
    if status:
        print(status, flush=True)
    
    if 'source' in audio_data and 'error' in audio_data:
        # update weights
        weights = update_weights(audio_data['source'], audio_data['error'], weights, 0.01)
        
        #process two inputs
        out_audio = filter_output(audio_data['source'], weights)
        outdata[:] = out_audio.reshape(1024, 1)

    else:
        outdata[:] = np.zeros_like(outdata)

start_idx = 0
def callback(outdata, frames, time, status):
        if status:
            print(status)
        global start_idx
        t = (start_idx + np.arange(frames)) / 44100
        t = t.reshape(-1, 1)
        outdata[:] = 0.5 * np.sin(2 * np.pi * 300 * t)
        start_idx += frames

def update_weights(src, err, w, lr):
    error = error_estimation(src, err, w)
    return np.add(w, lr * np.multiply(error, src))

def filter_output(src, w):
    return 100 * np.multiply(w, src)

def error_estimation(src, err, w):
    y = filter_output(src, w)
    return np.subtract(err, y)

print("Program START")
with sd.InputStream(callback=src_callback, blocksize=blocksize, channels=1, device=2), \
     sd.InputStream(callback=err_callback, blocksize=blocksize, channels=1, device=1), \
     sd.OutputStream(callback=out_callback, blocksize=blocksize, channels=1, device=5):

    time.sleep(0.5)
    while True:
        # reset plot
        fig.canvas.restore_region(bg)

        # roll plot data
        src_shift = len(audio_data['source'])
        src_plot = np.roll(src_plot, -src_shift,axis = 0)
        src_plot[-src_shift:] = audio_data['source']
        
        err_shift = len(audio_data['error'])
        err_plot = np.roll(err_plot, -err_shift,axis = 0)
        err_plot[-err_shift:] = audio_data['error']

        error_shift = len(audio_data['error'])
        error_plot = np.roll(error_plot, -error_shift,axis = 0)
        error_plot[-err_shift:] = error_estimation(audio_data['source'], audio_data['error'], weights)

        # update plot data
        src_ln.set_ydata(src_plot)
        err_ln.set_ydata(err_plot)
        error_ln.set_ydata(error_plot)

        # blit plot
        ax.draw_artist(src_ln)
        ax.draw_artist(err_ln)
        #ax.draw_artist(error_ln)
        
        fig.canvas.blit(fig.bbox)
        fig.canvas.flush_events()

        # check duration
        current_time = time.time()
        if current_time - start_time >= duration:
            end_time = current_time - start_time
            break

print("Elapsed time:", end_time)
print("Program END")
