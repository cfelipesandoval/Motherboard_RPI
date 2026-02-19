from serialMessenger import *
from dataIO import *

def main():
  print("Initializing Settings Serial Connection")
  sm = SerialMessenger(port=SETTINGS_PORT, baudrate=9600, timeout=5)

  print("Done Initializing Serial \n")
  
  collectTime = 1
  calibrationTime = collectTime / 10
  freq = 38.2
  clockFreq = 81.43
  decimation = 32
  
  # sm.send("N", 1, printResult = True)
  sm.send("C", clockFreq, printResult = True)
  # sm.send("S", clockFreq/(64*100), printResult = True)
  sm.send("D", decimation, printResult = True)
  sm.send("B", 0, printResult = True) # Set output to debug (1)
  
  sm.send("U", collectTime * 1000, printResult = True)
  sm.send("L", calibrationTime * 1000, printResult = True) 
  # sm.send("N", 0)
  sm.send("G", 50, printResult = True)
  sm.send("F", freq, printResult = True)
  # sm.send("I", 20, printResult=True)
  # sm.send("S", 0, printResult = True)

  time.sleep(1)

  # fileName = t + "F" + str(freq) + "C" + str(collectTime)
  
  # print(f"Output File Name: {fileName} \n")
  
  # need to fix file naming to account for extra second it takes to actually collect the data
  # outFiles = collectDayDataDual(sm, "COM21", "COM2", 1, 25, collectTime, [freq], collectSecond=0)
  # fileName = outFiles[0]
  # fileName = "D20260217T232859F20C2.bin"

  temp, t = sm.send("T", -1)
  fileName = t + "F" + str(freq) + "C" + str(collectTime)
  saveDataDual(sm, "COM21", "COM2", fileName, collectTime, t)
  # for fileName in outFiles:
  readDataDual(fileName, lower = -150, upper = -100, clockFreq=clockFreq, decimation=decimation, PLOT_FREQ = False)
  

if __name__ == '__main__':
  main()