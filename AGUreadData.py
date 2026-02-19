import serial
import numpy as np
import struct
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from serialMessenger import *
import os
import re
from dataIO import *

def main():
  # Read Data
  specArray = []
  freqsArray = []
  binsArray = []
  imArray = []
  
  clockFrequency = 114e6
  # centerFrequency = 30e6
  decimation = 16
  nfft = 8192
  
  
  collectTime = 2
  calibrationTime = collectTime / 10
  freq = 20
  clockFreq = 81.43
  decimation = 32
  
  fig, ax = plt.subplots(figsize=(10, 6))
  
  freqLow = 38.3e6
  freqHigh = 38.75e6
  freqLow = 20e6
  freqHigh = 60e6
  
  upper = -90
  lower = -120
  
  # freq = 38.2
  samples = 24
  collectTime = 2
  
  # This is to get the baseline noise from a small sample set, should remove
  # day = 3
  # t = "181910"
  # fileName = "D" + str(day) + "T" + t + "F" + str(freq) + "C" + str(collectTime)
  # noiseSpec, freqs, noiseBins, im = getSpectrogram(fileName, "D" + str(day) + "T" + t, freqLow, freqHigh, lower=lower, upper=upper, nfft=8192)
  
  # This is to read all the data starting at the day/time below
  t = "211700"
  tPrev = t
  day = 1
  
  minutesEvery = 5 * 4
  
  files = []
  
  for i in range(0,samples):
    print(t)
    if(int(t) - int(tPrev)) < 0:
      day += 1
    fileName = "D" + str(day) + "T" + t + "F" + str(freq) + "C" + str(collectTime)
    
    # spec, freqs, bins, im = getSpectrogram(fileName, "D" + str(day) + "T" + t, lower, upper)
    
    try:
      spec, freqs, bins, im = getSpectrogram(fileName, "D" + str(day) + "T" + t, freqLow, freqHigh, lower=lower, upper=upper, nfft=2048)
    except FileNotFoundError:
      print("File not found")
      # File does not exist → skip cleanly
      # i -= 1
      tPrev = t
      t = addMinutes(t, minutesEvery)
      continue
    
    # size = min(spec.shape[1], noiseSpec.shape[1])
    # specArray.append((spec[:,:size] - noiseSpec[:,:size]))
    specArray.append(spec)
    freqsArray.append(freqs)
    binsArray.append(bins)
    imArray.append(im)
    
    files.append(t)
    tPrev = t
    t = addMinutes(t, minutesEvery)

  specArray = np.concatenate(specArray, axis = 1)
  binsArray = np.concatenate(binsArray)
  freqs = np.array(freqsArray[0])

  
  ax.clear()
  # # # im = ax.imshow(10*np.log10(specArray), origin='lower', aspect='auto',
  # # #             extent=[0, samples, min(freqsArray[0]), max(freqsArray[0])],
  # # #             vmin = lower, vmax = upper)
  
  
  ax.pcolormesh(binsArray, freqs, 10*np.log10(specArray), shading = 'auto', vmin=lower, vmax=upper)
  
  plt.xlabel("Time (UTC)")
  # plt.ylabel("Frequency (Hz)")

  current_xticks = ax.get_xticks()
  # new_xticks = np.linspace(0, samples, samples)

  files = files[::10]

  formatted_labels = []
  for time_str in files:
    formatted_time = f"{time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}"
    formatted_labels.append(formatted_time)

  
  ax.set_xticklabels(formatted_labels)

  
  plt.title(f"1 Day Spectrogram Collected in Atlanta, GA Centered at 38.2 MHz")
  formatter = ticker.FuncFormatter(hz_to_mhz_formatter)
  ax.yaxis.set_major_formatter(formatter)
  
  cbar = plt.colorbar(im, ax=ax)
  cbar.set_label('Intensity (dBFS)') # Set a label for the colorbar
  plt.show()
  plt.colorbar(label="Intensity (dB)")
  
  # # ax.pcolormesh(noiseBins, freqs, 10*np.log10(noiseSpec), shading = 'auto', vmin=lower, vmax=upper)

  # # averages = np.mean(specArray, axis = 0)
  
  # def movingaverage(interval, window_size):
  #   window= np.ones(int(window_size))/float(window_size)
  #   return np.convolve(interval, window, 'same')

  # # with open("averagedata.bin", "wb") as f:
  # #   f.write(averages)
  
  # averages = np.fromfile("averagedata.bin")[8000:265000]
  # ax.clear()
  # # ax.plot(averages)
  # ax.plot(10*np.log10(movingaverage(movingaverage(-averages, 500),8000)))
  
  # plt.title("1 Day Derived Received Power Collected in Atlanta, GA")
  # plt.xlabel("Time (UTC)") 
  # plt.ylabel("Received Power (dB)")
  # plt.grid()
  
  # label_positions = np.linspace(0, len(averages), len(formatted_labels), endpoint=False)
  # ax.set_xticks(label_positions)

  # ax.set_xticks(label_positions)
  # ax.set_xticklabels(formatted_labels)
  
  plt.show()

if __name__ == "__main__":
  main()