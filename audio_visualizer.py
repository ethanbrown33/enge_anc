# Import packages
import numpy as np
import sounddevice as sd
import time
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import pywt
from scipy.optimize import curve_fit
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy import stats
import scipy

def denoise(y2):
  fft_signal = y2
  y3 = y2
  # y3 = []
  # count = 0
  # while(count < len(y2)):
  #   y3.append(y2[count])
  #   count += 1000
  def filter_signal(th):
    f_s = fft_filter(th)
    return np.real(np.fft.ifft(f_s))
  def fft_filter(perc):
      fft_signal = np.fft.fft(y3)
      fft_abs = np.abs(fft_signal)
      th=perc*(2*fft_abs[0:int(len(y3))]/((len(y3)))).max()
      fft_tof=fft_signal.copy()
      fft_tof_abs=np.abs(fft_tof)
      fft_tof[fft_tof_abs<=th]=0
      return fft_tof
  def fft_filter_amp(th):
      fft = np.fft.fft(y3)
      fft_tof=fft.copy()
      fft_tof_abs=np.abs(fft_tof)
      fft_tof_abs=2*fft_tof_abs/(len(fft_tof_abs)/2.)
      fft_tof_abs[fft_tof_abs<=th]=0
      return fft_tof_abs[0:int(len(fft_tof_abs)/2.)]
  th_list = np.linspace(0,1,5)
  th_list = th_list[0:len(th_list)-1]
  th_list = np.array([0, 0.25, 0.5, 0.75])
  th_list = np.linspace(0,0.02,1000)
  th_list = th_list[0:len(th_list)]
  p_values = []
  corr_values = []
  for t in th_list:
      filt_signal = filter_signal(t)
      res = stats.spearmanr(y3,y3-filt_signal)
      p_values.append(res.pvalue)
      corr_values.append(res.correlation) 
  th_opt = th_list[np.array(corr_values).argmin()]
  opt_signal = filter_signal(th_opt)
  return opt_signal

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
        # update the weights
        weights = update_weights(audio_data['source'], audio_data['error'], weights, 0.01)
        
        #process two inputs
        figure, axis = plt.subplots(2, 2) 
        axis[0,0].plot(audio_data['source']) #original audio
        axis[0,0].setTitle("Amplitude of the Audio Source")
        audio_data['source'] = denoise(audio_data['source'])
        axis[0,1].plot(audio_data['source']) #original audio post-denoising
        axis[0,1].setTitle("Amplitude of the Audio Source Post-Denoising")
        out_audio = filter_output(audio_data['source'], weights)
        axis[1,0].plot(out_audio) #audio after LMS
        axis[1,0].setTitle("Amplitude of the Audio Source Post-LMS")
        outdata[:] = out_audio.reshape(1024, 1)
        plt.show()

    else:
        outdata[:] = np.zeros_like(outdata)

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
     sd.OutputStream(callback=out_callback, blocksize=blocksize, channels=1, device=6):

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
