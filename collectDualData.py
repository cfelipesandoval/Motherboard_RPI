import time
import serial
import os
import numpy as np
import sys

LINUX = True

if LINUX:
  DIRECTORY = "/media/georgiatech.lf/T7 Shield/HF_Data" # For Raspi
else:
  DIRECTORY = os.getcwd() + "/output" # Save in current directory

def main():
  num = (sys.argv[1]) 
  collectPort = sys.argv[2]
  fileName = sys.argv[3] 
  collectTime = float(sys.argv[4])
  folder = sys.argv[5] 
  
  fileName = "N" + num + fileName
  
  # print(num, collectPort, fileName, collectTime, folder)
  
  ser = serial.Serial(collectPort, 115200, timeout = None)
  # ser.reset_input_buffer()
  chunk = None
  data = bytearray()
  
  # sys.stderr.write(num)
  print("1", flush = True)
  # sys.stderr.flush()

  while True:
    line = sys.stdin.readline()
    # if line == "":  # EOF (parent closed stdin)
    #     raise RuntimeError("stdin closed before START")
    if line.strip() == "1":
        break
  
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



if __name__ == '__main__':
  main()