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
  
  # Read number of valid IQ Samples
  samplesNum = ser.read(4)
  
  # Read the incoming data
  iqRaw = ser.read(arraySize)
  
  num_words = arraySize // 4
  data = struct.unpack('<' + 'I' * num_words, iqRaw)

  print("Size of Array: ", arraySize)
  print("Number of Samples in Array: ", samplesNum)
  print("Received Data: ", data)


if __name__ == '__main__':
  main()