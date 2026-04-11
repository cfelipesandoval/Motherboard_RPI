import serial
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from serialMessenger import *
import os
import re
from datetime import datetime, timedelta
import subprocess
import sys


LINUX = True


if LINUX:
  DIRECTORY = "/media/georgiatech.lf/T7 Shield/HF_Data" # For Raspi
else:
  DIRECTORY = os.getcwd() + "/output" # Save in current directory

# DIRECTORY = "E:/HF_Data" # for Windows

if LINUX:
  # USB Ports for Linux
  SETTINGS_PORT = "/dev/ttyACM1"
  COLLECT_PORT0 = "/dev/ttyACM0"   # e.g., "COM5" on Windows
  COLLECT_PORT1 = "/dev/ttyACM2"   # e.g., "COM5" on Windows
else:
  # USB Ports for Windows
  SETTINGS_PORT = "COM12"
  COLLECT_PORT0 = "COM21"
  COLLECT_PORT1 = "COM2"

PLOT_TIME = False
PLOT_FREQ = True

chunk = None
def saveDataDual(sm: SerialMessenger, ser1: str, ser2: str, fileName, collectTime, folder: str = ""):
  fileName = fileName + ".bin"
  p1 = subprocess.Popen([sys.executable, "collectDualData.py", '0', ser1, fileName, str(collectTime), folder], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
  p2 = subprocess.Popen([sys.executable, "collectDualData.py", '1', ser2, fileName, str(collectTime), folder], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
  
  line1 = ""
  line2 = ""

  while((line1 != "1")):
    line1 = p1.stdout.readline().strip()
  while((line2 != "1")):
    line2 = p2.stdout.readline().strip()

  # sm.send("M", -1, timeout = 2)

  p1.stdin.write("1\n"); 
  p1.stdin.flush()
  
  p2.stdin.write("1\n"); 
  p2.stdin.flush()

  print("Collecting Data")

  p1.wait()
  p2.wait()
  
  print("Done Collecting Data \n")
  
def collectDayDataDual(sm: SerialMessenger, ser1, ser2, intervalMinutes, maxMinutes, collectTime, freqs, collectSecond = 0):
  outFiles = []
  max_delta = timedelta(minutes = (maxMinutes - 1))
  temp, t = sm.send("T", -1)
  
  if((collectSecond - 2) < 0): pre = 60 - collectSecond - 2
  else: pre = collectSecond - 2
  
  while int(t[-2:]) != (pre):
    time.sleep(0.01)
    temp, t = sm.send("T", -1, timeout = None)

  initialTime = t
  
  print("Locked in")
  
  lastTime = None
  if((collectSecond - 2) < 0): pre = 60 - collectSecond - 1
  else: pre = collectSecond - 1
  while True:
    time.sleep(0.01)
    temp, t = sm.send("T", -1)
    while int(t[-2:]) != (pre):
      time.sleep(0.01)
      temp, t = sm.send("T", -1, timeout = None)

    if(lastTime == t):
      continue # Make sure you don't collect more than once every collect period
    
    lastTime = t
    
    if((((int(t[-4:-2]) - int(initialTime[-4:-2])) - 1) % intervalMinutes == 0)):
      # try:
      #   sm.send("M","", timeout=5)
      # except TimeoutError:
      #   continue
      
      print("Saving " + t)
      
      temp, t = sm.send("T", -1)
      
      
      # fileName = t + "F" + str(freqs) + "C" + str(collectTime)
      # saveDataDual(sm, ser1, ser2, fileName, collectTime, t)
      
      # outFiles.append(fileName)
      
      for f in freqs:
        fileName = t + "F" + str(f) + "C" + str(collectTime)
        sm.send("F", f, printResult=True)
        time.sleep(0.1)
        saveDataDual(sm, ser1, ser2, fileName, collectTime, t)
        outFiles.append(fileName)

      ## gotta fix this
      # t0 = datetime.strptime(
      # re.search(r"D(\d{8}T\d{6})", initialTime).group(1), "%Y%m%dT%H%M%S")

      # # Extract and parse current
      # t1 = datetime.strptime(
      #     re.search(r"D(\d{8}T\d{6})", t).group(1), "%Y%m%dT%H%M%S")

      # if (t1 - t0) >= max_delta:
      #   return outFiles


def saveData(ser: serial.Serial, sm: SerialMessenger, fileName, collectTime, folder: str = ""):
  chunk = None
  data = bytearray()
  # ser.reset_input_buffer()
  
  # while((time.time() - start) < collectTime):
  # sm.send_ctrl_c()
  # sm.send("D", 16)
  # sm.send("C", 114)
  sm.send_only("U", collectTime * 1000)
  sm.send("M", -1)
  
  # exe = r".\test.exe"
  # com_port = "COM4"
  # baud = "200000000"
  # outfile = "output\\" + folder + "\\" + fileName + ""
  # duration = str(1.5 * collectTime)

  # cmd = [exe, com_port, baud, outfile, duration]
  
  # print(cmd)
  
  # '''
  # maybe test it to send int16 instead of uint32
  # and see if speed gets higher??
  # '''
  
  # p = subprocess.Popen(cmd)
  
  # while p.poll() is None:
  #   print("Logger running...")
  #   time.sleep(1)
  
  start = time.time()
  while((time.time() - start) < 2 * collectTime):
    # chunk = ser.read(PACKET_N * BYTES_PER_PACKET) # Maybe I should change the amount to collect
    chunk = ser.read(ser.in_waiting)
    while chunk is None:
      chunk = ser.read(ser.in_waiting)

    data.extend(chunk)
    # continue

  # print(ser.in_waiting)
  
  data = np.frombuffer(data, dtype = np.int16)
  # print(bin(data))
  # data = data.astype(np.float16) / 32768 # Maybe remove / 32768
  # data = (data >> 16).astype(np.int16)
  
  outFolder = DIRECTORY + "/" + folder
  
  filePath = os.path.join(outFolder, fileName)
  os.makedirs(outFolder, exist_ok=True)
  
  with open(filePath, 'wb') as f:
    f.write(data.astype('int16').tobytes())

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

def hz_to_mhz_formatter(x, pos):
  return f'{x/1e6:g} MHz' # Divides by 1e6 and formats as a general number

def hz_to_khz_formatter(x, pos):
  return f'{x/1e3:g} kHz' # Divides by 1e6 and formats as a general number

def readDataDual(fileName, lower = -100, upper = -30, clockFreq = 95, decimation = 16, PLOT_TIME = True, PLOT_FREQ = True, PLOT_DIFFERENCE = False):
  # folder = 'T160550'
  # fileName = 'T160550F18.8C3'
  fileName = fileName + ".bin"
  folder = re.search(r"D\d{8}T\d{6}", fileName).group()
  outFolder = "output" + "/" + folder
  
  # outFolder = "" + folder
  
  filePath = os.path.join(outFolder, "N0" + fileName)
  data0 = np.fromfile(filePath, dtype = np.int16) # you can use this to send
  filePath = os.path.join(outFolder, "N1" + fileName)
  data1 = np.fromfile(filePath, dtype = np.int16) # you can use this to send

  # plt.plot((data0 & 1))
  # plt.plot((data1 & 1))
  # plt.show()
  
  k = 50000
  
  # print(f"Data before: {len(data0)}")
  # ind = (data0 & 1) == 1
  # idx = np.argmax(ind) if np.any(ind) else None
  # start = max(0, idx - k)
  # ind[start:idx] = True
  
  # data0 = data0[ind]
  data0 = data0 & ~(1)
  print(f"Data after: {len(data0)}")
  
  print(f"Data before: {len(data1)}")
  
  # ind = (data1 & 1) == 1
  # idx = np.argmax(ind) if np.any(ind) else None
  # start = max(0, idx - k)
  # # ind[start:idx] = True
  # data1 = data1[ind]
  # ind = (data0 & 1) == 1
  data1 = data1 & ~(1)
  # print(f"Data after: {len(data1)}")
  
  # data0 = (data0 >> 16).astype(np.int16).astype(np.float16) / 32768
  # data1 = (data1 >> 16).astype(np.int16).astype(np.float16) / 32768
  
  data0 = (data0).astype(np.float16) / 32768
  data1 = (data1).astype(np.float16) / 32768
  
  print(len(data0), len(data1))
  fileName = fileName.replace(".bin",  "")
  pattern = r"D(?P<date>\d{8})T(?P<time>[^FC]+)F(?P<freq>[^C]+)C(?P<collect>.+)"
  match = re.match(pattern, fileName)

  if not match:
      raise ValueError("Filename format not recognized")

  time_val = match.group("time")
  freq_val = match.group("freq")
  collect_val = match.group("collect")
  
  collectTime = float(collect_val)
  
  clockFrequency = clockFreq * 1e6
  centerFrequency = float(freq_val) * 1e6
  nfft = 8192

  samples0 = data0
  samples1 = data1
  
  # print(samples0[0])
  # print(samples1[0])
  
  val = min(len(samples0), len(samples1))
  
  samples0 = samples0[:val]
  samples1 = samples1[:val]
  
  collectTime = len(samples0) / (2 * clockFrequency / decimation)
  print(f'collect time: {collectTime}')
  
  # This need to be fixed a different way
  # i.e. the amplitude difference between both channels
  
  # scale = max(abs(samples1)) / max(abs(samples0))
  # samples0 = samples0 * scale
  # samples1 = samples1 
  
  # # ind = int(np.floor(0.2 * 2 * clockFreq / decimation * 1e6))
  # calibration0 = max(abs(samples0))
  # calibration1 = max(abs(samples1))
  # samples0 = samples0 / calibration0 * 250/2000
  # samples1 = samples1 / calibration1 * 250/2000
  
  # print(calibration0, calibration1)
  # samples0 = samples0[]
  # samples1 = samples1[]

  # samples0 = samples0[ind]
  # samples1 = samples1[ind]

  PLOT_PHASE = False
  plot_phase = False

  # dphi = dphi[:1000000]
  print("done")
  # plt.figure(0)
  # plt.plot(samples0)
  # plt.plot(samples0, 'bx')
  # plt.plot(upsampleIndex, y)
  
  skip = 1000
  if(PLOT_TIME):
    plt.plot(samples0[::skip])
    plt.plot(samples1[::skip])
  
  if(PLOT_DIFFERENCE):
    plt.plot((samples0[::skip] - samples1[::skip]))

  # if(plot_phase): plt.plot(dphi)
  
  # plt.plot(dphi0)
  # plt.plot(dphi1)
  
  plt.legend(["X-Arm", "Y-Arm", "X - Y"])
  # plt.title("Phase of measured data")
  plt.title(f"Both Channels Superimposed and Their Difference with Fc = {freq_val} MHz and Bandwidth = {clockFreq/decimation} MHz")
  plt.xlabel(f"Sample index")
  # plt.ylabel("Phase (rad)")
  plt.ylabel("Value")
  
  # plt.show()
  
  # readData("D0" + fileName, outFolder, clockFreq, decimation, 20, -0)
  # readData("D1" + fileName, outFolder, clockFreq, decimation, 20, -0)
  
  
  
  
  # ind = round(0.01 * 2 * clockFreq / decimation * 1e6) 
  # samples0 = samples0[ind:]
  # samples1 = samples1[ind:]
  
  if(PLOT_FREQ):
    i = 0
    channel = ["X", "Y"]
    
    for data in [samples0, samples1]:
      fig, ax = plt.subplots(figsize=(10, 6))
      # spec, freqs, t, im = ax.specgram(data * 1000, NFFT = nfft, Fs = 2 * clockFrequency/(decimation), Fc = centerFrequency - clockFrequency / (decimation * 2), xextent = (0,collectTime), scale='dB', vmin=lower, vmax = upper) # 
      spec, freqs, bins, im = plt.specgram(data, NFFT = nfft, Fs = clockFrequency, scale='linear', vmin=lower,vmax=upper) # 

      freqs /= decimation / 2
      freqs += centerFrequency - clockFrequency / (decimation * 2)
      
      ax.imshow(10*np.log10(spec), origin='lower', aspect='auto',
                  extent=[0, collectTime, min(freqs), max(freqs)],
                  vmin = lower, vmax = upper)
      plt.xlabel("Time (s)")
      # plt.ylabel("Frequency (Hz)")

      current_xticks = ax.get_xticks()
      new_xticks = np.linspace(0,collectTime,len(current_xticks))
      ax.set_xticks(new_xticks)
      
      plt.title(f"Channel {channel[i]} with Fc = {freq_val} MHz and Bandwidth = {clockFreq/decimation} MHz")
      formatter = ticker.FuncFormatter(hz_to_mhz_formatter)

      # Set the major formatter for the y-axis
      ax.yaxis.set_major_formatter(formatter)
      
      cbar = plt.colorbar(im)
      cbar.set_label('Intensity') # Set a label for the colorbar
      i += 1
      
      # plt.colorbar(label="Intensity (dB)")
  plt.show()


def readDataDualFFT( fileName, lower_db = -120, upper_db = 0, clockFreq = 95, decimation = 16, PLOT_TIME = True, PLOT_FREQ = True, PLOT_DIFFERENCE = False, APPLY_WINDOW = True, REMOVE_DC = True):
  fileName = fileName + ".bin"
  folder = re.search(r"D\d{8}T\d{6}", fileName).group()
  outFolder = os.path.join("output", folder)

  filePath0 = os.path.join(outFolder, "N0" + fileName)
  filePath1 = os.path.join(outFolder, "N1" + fileName)

  data0 = np.fromfile(filePath0, dtype=np.int16)
  data1 = np.fromfile(filePath1, dtype=np.int16)

  print(f"Data after ch0: {len(data0)}")
  print(f"Data after ch1: {len(data1)}")

  data0 = data0.astype(np.float32) / 32768.0
  data1 = data1.astype(np.float32) / 32768.0

  data0 = data0[:15000000]
  data1 = data1[:15000000]
  
  
  print(len(data0), len(data1))

  baseName = fileName.replace(".bin", "")
  pattern = r"D(?P<date>\d{8})T(?P<time>[^FC]+)F(?P<freq>[^C]+)C(?P<collect>.+)"
  match = re.match(pattern, baseName)

  if not match:
    raise ValueError("Filename format not recognized")

  freq_val = match.group("freq")
  collect_val = match.group("collect")

  clockFrequency = clockFreq * 1e6
  centerFrequency = float(freq_val) * 1e6

  samples0 = data0
  samples1 = data1

  val = min(len(samples0), len(samples1))
  samples0 = samples0[:val]
  samples1 = samples1[:val]

  # fs = 2 * clockFrequency / decimation
  fs = clockFrequency
  collectTime = len(samples0) / (2 * clockFrequency / decimation)
  print(f"Collect time: {collectTime}")

  # Match original amplitude balancing logic
  # max0 = np.max(np.abs(samples0))
  # max1 = np.max(np.abs(samples1))
  # if max0 > 0 and max1 > 0:
  #   scale = max1 / max0
  #   samples0 = samples0 * scale

  # calibration0 = np.max(np.abs(samples0))
  # calibration1 = np.max(np.abs(samples1))

  # if calibration0 > 0:
  #   samples0 = samples0 / calibration0 * 250 / 2000
  # if calibration1 > 0:
  #   samples1 = samples1 / calibration1 * 250 / 2000

  # print(f"Calibration0: {calibration0}, Calibration1: {calibration1}")
  print("Done")

  skip = 1000

  if PLOT_TIME:
    plt.figure(figsize=(10, 5))
    plt.plot(samples0[::skip], label="X-Arm")
    plt.plot(samples1[::skip], label="Y-Arm")

    if PLOT_DIFFERENCE:
      plt.plot(samples0[::skip] - samples1[::skip], label="X - Y")

    plt.legend()
    plt.title(
      f"Both Channels Superimposed"
      f" with Fc = {freq_val} MHz and Bandwidth = {clockFreq / decimation} MHz"
    )
    plt.xlabel("Sample index")
    plt.ylabel("Value")

  if PLOT_FREQ:
    channels = [("X", samples0), ("Y", samples1)]

    for ch_name, data in channels:
      x = data.copy()

      # if REMOVE_DC:
      #   x = x - np.mean(x)

      N = len(x)

      if APPLY_WINDOW:
        window = np.hanning(N)
        x_fft_in = x * window
        coherent_gain = np.mean(window)
      else:
        x_fft_in = x
        coherent_gain = 1.0

      N = round(N / 8192)
      
      fft_vals = np.fft.rfft(x_fft_in, n = N)
      freqs = np.fft.rfftfreq(N, d=decimation/(2*fs))
      # freqs /= decimation / 2
      # freqs += centerFrequency - clockFrequency / (decimation * 2)
      
      freqs_rf = freqs+centerFrequency-clockFrequency/(2*decimation)

      # Shift to actual RF axis like your original spectrogram logic
      # freqs_rf = freqs + centerFrequency


      # Normalize magnitude
      mag = np.abs(fft_vals) / (N * coherent_gain)

      # Avoid log of zero
      mag_db = 20 * np.log10(np.maximum(mag, 1e-15))

      fig, ax = plt.subplots(figsize=(10, 6))
      ax.plot(freqs_rf, mag_db)
      # ax.plot(freqs_rf, mag)

      ax.set_title(
        f"Channel {ch_name} FFT"
        f" with Fc = {freq_val} MHz and Bandwidth = {clockFreq / decimation} MHz"
      )
      ax.set_xlabel("Frequency (MHz)")
      ax.set_ylabel("Magnitude (dB)")
      # ax.set_ylim(lower_db, upper_db)

      formatter = ticker.FuncFormatter(hz_to_mhz_formatter)
      ax.xaxis.set_major_formatter(formatter)
      ax.grid(True)
  # plt.show()



def readData(fileName, folder, clockFreq = 95, dec = 16, upper = 0, lower = -120):
  outFolder = folder
  # outFolder = "" + folder
  filePath = os.path.join(outFolder, fileName)
  
  data = np.fromfile(filePath, dtype = np.uint32) # you can use this to send
  
  # plt.plot((data & (1)))
  # plt.show()
  
  # print(np.sum(((data >> 13) & 1)))
  # print(f"Data before: {len(data)}")
  # data = data[(data & 1) == 1]
  # data = data & ~(1)
  # print(f"Data after: {len(data)}")
  
  # data = (data).astype(np.float16) / 32768
  (data >> 16).astype(np.int16).astype(np.float16) / 32768

  # data = data[np.nonzero(data)].astype(np.float16) / 32768
  
  
  # print(len(np.nonzero(data)))
  print(len(data))
  
  # pattern = r"T(?P<time>[^F^C]+)F(?P<freq>[^C]+)C(?P<collect>.+)"
  pattern = r"(?:D(?P<device>\d+))?T(?P<time>[^FC]+)F(?P<freq>[^C]+)C(?P<collect>.+)"
  match = re.match(pattern, fileName)

  if not match:
      raise ValueError("Filename format not recognized")

  time_val = match.group("time")
  freq_val = match.group("freq")
  collect_val = match.group("collect")
  
  collectTime = float(collect_val)
  
  
  clockFrequency = clockFreq * 1e6
  centerFrequency = float(freq_val) * 1e6
  decimation = dec
  nfft = 8192
  
  collectTime = len(data) / (clockFrequency / decimation)
  print(f'collect time: {collectTime}')
  
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
               extent=[0, collectTime, min(freqs), max(freqs)],
               vmin = lower, vmax = upper)
    plt.xlabel("Time (s)")
    # plt.ylabel("Frequency (Hz)")

    current_xticks = ax.get_xticks()
    new_xticks = np.linspace(0,collectTime,len(current_xticks))
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

def getSpectrogram(fileName, folder, freqLow, freqHigh, lower = -120, upper = 0, nfft = 1024):
  outFolder = DIRECTORY + "/" + folder
  filePath = os.path.join(outFolder, fileName)
  
  data = np.fromfile(filePath, dtype = np.uint32) # you can use this to send
  data = (data >> 16).astype(np.int16).astype(np.float16) / 32768
  
  # pattern = r"T(?P<time>[^F^C]+)F(?P<freq>[^C]+)C(?P<collect>.+)"
  pattern = r"D(?P<day>\d+)T(?P<time>[^FC]+)F(?P<freq>[^C]+)C(?P<collect>.+)"

  match = re.match(pattern, fileName)

  if not match:
      raise ValueError("Filename format not recognized")

  time_val = match.group("time")
  freq_val = match.group("freq")
  collect_val = match.group("collect")
  
  collectTime = float(collect_val) / 10
  
  clockFrequency = 114e6
  centerFrequency = float(freq_val) * 1e6
  decimation = 16
  # nfft = 1024
  
  samples = int(collectTime * clockFrequency / decimation * 2)  # didiving by 10 to get a tenth of the data
  data = data[:samples]

  # if PLOT_TIME:
  #   plt.figure(0)
  #   plt.plot(data)
  # if PLOT_FREQ:
    # fig, ax = plt.subplots(figsize=(10, 6))
    # spec, freqs, t, im = ax.specgram(data * 1000, NFFT = nfft, Fs = 2 * clockFrequency/(decimation), Fc = centerFrequency - clockFrequency / (decimation * 2), xextent = (0,collectTime), scale='dB', vmin=lower, vmax = upper) # 
  spec, freqs, bins, im = plt.specgram(data, NFFT = nfft, Fs = clockFrequency, scale='dB', vmin = lower, vmax = upper) # 

  freqs /= decimation / 2
  freqs += centerFrequency - clockFrequency / (decimation * 2)
  
  indices = np.where((freqs > freqLow) & (freqs < freqHigh))
  
  return spec[indices], freqs[indices], bins, im

def addMinutes(utc_str, minutes):
  """
  utc_str: string
  seconds: integer seconds to add (can be negative)

  Returns new string "HHMMSS" after proper rollover.
  """
  # Parse as a dummy date + the given time
  t = datetime.strptime(utc_str, "%H%M%S")

  # Add the offset
  t_new = t + timedelta(minutes=minutes)

  # Convert back to HHMMSS
  return t_new.strftime("%H%M%S")
