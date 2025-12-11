import serial
# import RPi.GPIO as GPIO
# from RealTimeSpectogram import *
import numpy as np
import time
import struct
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from serialMessenger import *
import os
import re

DIRECTORY = "/media/georgiatech.lf/T7 Shield/HF_Data"
# DIRECTORY = "HF_Data"

COLLECT_PORT = "/dev/ttyACM0"   # e.g., "COM5" on Windows
SETTINGS_PORT = "/dev/ttyACM1"
BAUD = 4000000
# FS   = 1e6       # sample rate (Hz) of real input
PACKET_N = 8192
BYTES_PER_PACKET = 4

PLOT_TIME = False
PLOT_FREQ = True

# collectTime = 60

chunk = None
# data = np.empty(0)

def hz_to_mhz_formatter(x, pos):
  return f'{x/1e6:g} MHz' # Divides by 1e6 and formats as a general number

def hz_to_khz_formatter(x, pos):
  return f'{x/1e3:g} kHz' # Divides by 1e6 and formats as a general number

def saveData(ser, fileName, collectTime, folder: str = ""):
  chunk = None
  data = bytearray()
  # ser.reset_input_buffer()
  start = time.time()
  while((time.time() - start) < collectTime):
    chunk = ser.read(PACKET_N * BYTES_PER_PACKET) # Maybe I should change the amount to collect
    
    while chunk is None:
      chunk = ser.read(PACKET_N * BYTES_PER_PACKET)

    data.extend(chunk)

  data = np.frombuffer(data, dtype = np.uint32)
  # data = data.astype(np.float16) / 32768 # Maybe remove / 32768
  # data = (data >> 16).astype(np.int16)
  
  outFolder = DIRECTORY + "/" + folder
  
  filePath = os.path.join(outFolder, fileName)
  os.makedirs(outFolder, exist_ok=True)
  
  with open(filePath, 'wb') as f:
    f.write(data.astype('uint32').tobytes())
  

def main():
  fileName = "test_data_30M60s2.bin"
  collectTime = 10
  
  ser = serial.Serial(COLLECT_PORT, BAUD)
  ser.reset_input_buffer()
  # time.sleep(2)
  print("Collecting Data")
  
  saveData(ser, fileName, collectTime)
  
  readData(fileName, collectTime)


def getSpectrogram(fileName, folder, upper = 0, lower = -120):
  outFolder = DIRECTORY + "/" + folder
  filePath = os.path.join(outFolder, fileName)
  
  data = np.fromfile(filePath, dtype = np.uint32) # you can use this to send
  data = (data >> 16).astype(np.int16).astype(np.float16) / 32768
  
  pattern = r"T(?P<time>[^F^C]+)F(?P<freq>[^C]+)C(?P<collect>.+)"
  match = re.match(pattern, fileName)

  if not match:
      raise ValueError("Filename format not recognized")

  time_val = match.group("time")
  freq_val = match.group("freq")
  collect_val = match.group("collect")
  
  collectTime = float(collect_val)
  
  clockFrequency = 114e6
  centerFrequency = float(freq_val) * 1e6
  decimation = 16
  nfft = 8192
  
  samples = int(collectTime * clockFrequency / decimation * 2)
  
  data = data[:samples]

  # if PLOT_TIME:
  #   plt.figure(0)
  #   plt.plot(data)
  # if PLOT_FREQ:
    # fig, ax = plt.subplots(figsize=(10, 6))
    # spec, freqs, t, im = ax.specgram(data * 1000, NFFT = nfft, Fs = 2 * clockFrequency/(decimation), Fc = centerFrequency - clockFrequency / (decimation * 2), xextent = (0,collectTime), scale='dB', vmin=lower, vmax = upper) # 
  spec, freqs, bins, im = plt.specgram(data, NFFT = nfft, Fs = clockFrequency, scale='linear', vmin=lower,vmax=upper) # 

  freqs /= decimation / 2
  freqs += centerFrequency - clockFrequency / (decimation * 2)
  
  return spec, freqs, bins, im

def readData(fileName, folder, upper = 0, lower = -120):
  outFolder = "output" + "/" + folder
  filePath = os.path.join(outFolder, fileName)
  
  data = np.fromfile(filePath, dtype = np.uint32) # you can use this to send
  data = (data >> 16).astype(np.int16).astype(np.float16) / 32768
  
  pattern = r"T(?P<time>[^F^C]+)F(?P<freq>[^C]+)C(?P<collect>.+)"
  match = re.match(pattern, fileName)

  if not match:
      raise ValueError("Filename format not recognized")

  time_val = match.group("time")
  freq_val = match.group("freq")
  collect_val = match.group("collect")
  
  collectTime = float(collect_val)
  
  clockFrequency = 114e6
  centerFrequency = float(freq_val) * 1e6
  decimation = 16
  nfft = 8192
  
  if PLOT_TIME:
    plt.figure(0)
    plt.plot(data)
  if PLOT_FREQ:
    fig, ax = plt.subplots(figsize=(10, 6))
    # spec, freqs, t, im = ax.specgram(data * 1000, NFFT = nfft, Fs = 2 * clockFrequency/(decimation), Fc = centerFrequency - clockFrequency / (decimation * 2), xextent = (0,collectTime), scale='dB', vmin=lower, vmax = upper) # 
    spec, freqs, bins, im = plt.specgram(data, NFFT = nfft, Fs = clockFrequency, scale='linear', vmin=lower,vmax=upper) # 

    freqs /= decimation / 2
    freqs += centerFrequency - clockFrequency / (decimation * 2)
    
    ax.imshow(10*np.log10(spec), origin='lower', aspect='auto',
               extent=[0, collectTime * 32, min(freqs), max(freqs)],
               vmin = lower, vmax = upper)
    plt.xlabel("Time (s)")
    # plt.ylabel("Frequency (Hz)")

    current_xticks = ax.get_xticks()
    new_xticks = np.linspace(0,collectTime * 32,len(current_xticks))
    ax.set_xticks(new_xticks)
    
    plt.title(f"Fc = {freq_val} MHz")
    formatter = ticker.FuncFormatter(hz_to_mhz_formatter)

    # Set the major formatter for the y-axis
    ax.yaxis.set_major_formatter(formatter)
    
    cbar = plt.colorbar(im)
    cbar.set_label('Intensity') # Set a label for the colorbar
    
    
    # plt.colorbar(label="Intensity (dB)")
    plt.show()
  
  # plt.show()
  
def collectDayData(ser, sm, intervalMinutes, maxMinutes, collectTime, freqs, collectSecond = 0):
  temp, t = sm.send("T", -1)
  
  if((collectSecond - 2) < 0): pre = 60 - collectSecond - 2
  else: pre = collectSecond - 2
  
  while int(t[-2:]) != (pre):
    time.sleep(0.01)
    temp, t = sm.send("T", -1, timeout = None)
  
  initialTime = t
  
  print("Locked in")
  
  lastTime = None
  
  while True:
    time.sleep(0.01)
    temp, t = sm.send("T", -1)
    while int(t[-2:]) != (collectSecond):
      time.sleep(0.01)
      temp, t = sm.send("T", -1, timeout = None)

    if(lastTime == t):
      continue # Make sure you don't colelct more than once every collect period
    
    lastTime = t
    
    if((((int(t[-4:-2]) - int(initialTime[-4:-2])) - 1) % intervalMinutes == 0)):
      try:
        sm.send("M","", timeout=5)
      except TimeoutError:
        continue
      
      print("Saving " + t)
      
      temp, t = sm.send("T", -1)
      
      for i in range(len(freqs)):
        fileName = "T" + t + "F" + str(freqs[i]) + "C" + str(collectTime)
        sm.send("F", freqs[i])
        saveData(ser, fileName, collectTime, "T" + t)
    
    # if((int(t[-4:-2]) - int(initialTime[-4:-2])) >= maxMinutes):
    #   return  

if __name__ == '__main__':
  # main()
  
  print("Initializing Serial for Teensy")
  ser = serial.Serial(COLLECT_PORT, BAUD, timeout = None)
  ser.reset_input_buffer()
  
  print("Initializing Serial")
  sm = SerialMessenger(port=SETTINGS_PORT, baudrate=9600, timeout=2)
  
  collectTime = 0.1
  
  # freq = 30
  
  # temp, t = sm.send("T", -1, timeout = None)

  # t = "224025"
  
  # fileName = "T" + t + "F" + str(freq) + "C" + str(collectTime)
  
  # sm.send("F", freq)
  
  # saveData(ser, fileName, collectTime, "T" + t)
  # readData(fileName, "T" + t, -80, -120)
  
  # print("Collecting Data")
  
  # temp, t = sm.send("T", -1)
  
  # for f in range (20,40):
  #   fileName = "T" + t + "F" + str(f) + "C" + str(collectTime)
  #   sm.send("F", f)
  #   saveData(ser, fileName, collectTime)
  # print("Done Collecting")
  
  # t = 183736
  
  # Collect day data
  freqs = [30, 35, 38.2, 40, 45, 50]
  
  collectDayData(ser, sm, 1, 120, 0.1, freqs)
  
  print("Done Collecting")
  
  # Read Data
  '''
  specArray = [] 
  freqsArray = [] 
  binsArray = [] 
  imArray = []
  
  clockFrequency = 114e6
  # centerFrequency = 30e6
  decimation = 16
  nfft = 8192
  fig, ax = plt.subplots(figsize=(10, 6))
  
  upper = -30
  lower = -120
  
  t = "025100"
  freq = 50
  samples = 32
  
  for i in range(0,samples):
    print(t)
    fileName = "T" + t + "F" + str(freq) + "C" + str(collectTime)
    
    spec, freqs, bins, im = getSpectrogram(fileName, "T" + t, lower, upper)
    
    specArray.append(spec)
    freqsArray.append(freqs)
    binsArray.append(bins)
    imArray.append(im)
    
    t = add_seconds(t, 5)

  specArray = np.concatenate(specArray, axis = 1)
  # specArray = np.flipud(specArray)
  freqs = np.array(freqsArray[0])
  
  print("freqs[0], freqs[-1] =", freqs[0], freqs[-1])
  print("min, max =", freqs.min(), freqs.max())
  ax.clear()
  im = ax.imshow(10*np.log10(specArray), origin='lower', aspect='auto',
              extent=[0, samples, min(freqsArray[0]), max(freqsArray[0])],
              vmin = lower, vmax = upper)
  plt.xlabel("Time (s)")
  # plt.ylabel("Frequency (Hz)")

  current_xticks = ax.get_xticks()
  # new_xticks = np.linspace(0, samples, samples)
  new_xticks = range(0,samples)
  ax.set_xticks(new_xticks)
  
  plt.title(f"Fc = 30 MHz")
  formatter = ticker.FuncFormatter(hz_to_mhz_formatter)

  # Set the major formatter for the y-axis
  ax.yaxis.set_major_formatter(formatter)
  
  cbar = plt.colorbar(im, ax=ax)
  cbar.set_label('Intensity') # Set a label for the colorbar
  
  # plt.colorbar(label="Intensity (dB)")
  plt.show()
  '''
  
  # for f in range (20,30):
  #   fileName = "T" + str(t) + "F" + str(f) + "C" + str(collectTime)
  #   readData(fileName, -50, -100)
  
  # folder = "T215500"
  # fileName = folder + "F" + str(30) + "C" + str(collectTime)
  
  
  