# Import libraries/modules
import pyaudio
import matplotlib.pyplot as plt
import numpy as np

# PyAudio input parameters
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
RECORD_SECONDS = 10
md = 10000 # max display frames (width of plot)

# declare initial axis
x = list(range(md))
y = [0] * md

# initial plot with dimensions
fig, ax = plt.subplots()
(ln,) = ax.plot(x, y, animated=True)
plt.ylim(-10000, 10000)
plt.xlim(0, md)

# show plot
plt.show(block=False)
plt.pause(0.01)

# set up blitting
bg = fig.canvas.copy_from_bbox(fig.bbox)
ax.draw_artist(ln)
fig.canvas.blit(fig.bbox)

# start audio stream
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index = 1,
                frames_per_buffer=CHUNK)

# recording loop
print('start recording')
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    # reset plot
    fig.canvas.restore_region(bg)  
  
    # read CHUNK most recent frames
    data = stream.read(CHUNK)
    amp = np.frombuffer(np.array(data), np.int16)

    # add new frames to y axis
    y.extend(amp.tolist())

    # discard old frames that no longer fit on plot
    y = y[-md:]

    # update y-axis data
    ln.set_ydata(y)

    # blit plot
    ax.draw_artist(ln)
    fig.canvas.blit(fig.bbox)
    fig.canvas.flush_events()

# close stream
print('done recording')
stream.stop_stream()
stream.close()
p.terminate()
