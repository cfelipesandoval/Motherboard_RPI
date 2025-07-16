import serial
import RPi.GPIO as GPIO
import time
import struct

startPin = 17
readyPin = 18

def main():
  GPIO.setmode(GPIO.BCM)
  GPIO.setup(startPin, GPIO.OUT)
  GPIO.setup(readyPin, GPIO.IN)
  
  input("Press Enter to Start ")
  
  GPIO.output(startPin, GPIO.HIGH)
  time.sleep(1) # Keep it high for 1 second
  GPIO.output(startPin, GPIO.LOW)
  
  ser = serial.Serial('/dev/ttyACM0', 1152000)
  
  while(not GPIO.input(readyPin)):
    continue
  
  # Tell MCU to start sending the necessary data
  ser.write(b'S')
  
  
  ## Start Reading the Data 
  ## (Probably need to add some sort of check for received bytes or such)  
  
  # Read size of incoming IQ array
  arraySize = ser.read(4)
  (arraySize,) = struct.unpack('<I', arraySize)
  
  # Read number of valid IQ Samples
  samplesNum = ser.read(4)
  
  # Read the incoming data
  iqRaw = ser.read(arraySize)
  
  num_words = arraySize // 4
  data = struct.unpack('<' + 'I' * num_words, iqRaw)

  print("Size of Array: ", arraySize)
  print("Number of Samples in Array: ", samplesNum)
  print("Received Data: \n")
  for i, val in enumerate(data):
    print(f"{i:03}: {val:02b}")



import struct
import numpy as np

def write_variable_to_file(filename, varname, data, dtype_code):
  """
  Append a variable to a binary file with structured metadata.

  Parameters:
  - filename: string, output file name
  - varname: string, variable name
  - data: list or NumPy array of values
  - dtype_code: int
      - 0 = float64 ("double")
      - 10 = float32 ("single")
      - 30 = int16
      - 50 = uint8 ("uchar")
  """

  # Map dtype code to NumPy type
  if dtype_code == 0:
    np_dtype = np.float64
  elif dtype_code == 10:
    np_dtype = np.float32
  elif dtype_code == 30:
    np_dtype = np.int16
  elif dtype_code == 50:
    np_dtype = np.uint8
  else:
    raise ValueError("Unsupported dtype code")

  data = np.array(data, dtype=np_dtype)
  data_length = len(data)

  with open(filename, 'ab') as f:  # Append mode (creates if needed)
    f.write(struct.pack('<i', dtype_code))          # 1. Type code
    f.write(struct.pack('<i', data_length))         # 2. Data length
    f.write(struct.pack('<i', 1))                   # 3. Columns = 1
    f.write(struct.pack('<i', 0))                   # 4. Not complex
    f.write(struct.pack('<i', len(varname)))        # 5. Name length
    f.write(varname.encode('ascii') + b'\x00')      # 6. Name + null terminator
    f.write(data.tobytes())                         # 7. Raw binary data



if __name__ == '__main__':
  main()