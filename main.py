from serialMessenger import *
from dataIO import *

def main():
  collectTime = 5
  calibrationTime = collectTime / 100
  freq = 28.2
  clockFreq = 71.25
  decimation = 32
  
  print("Initializing Settings Serial Connection")
  sm = SerialMessenger(port=SETTINGS_PORT, baudrate=9600, timeout=5)
  print("Done Initializing Serial \n")
  
  sm.send("C", clockFreq, printResult = True)
  sm.send("D", decimation, printResult = True)
  # sm.send("B", 0, printResult = True) # Set output to debug (1)
  
  sm.send("U", collectTime * 1000, printResult = True)
  sm.send("L", calibrationTime * 1000, printResult = True) 
  # sm.send("N", 0)
  sm.send("G", 50, printResult = True)
  # sm.send("F", freq, printResult = True)

  time.sleep(1)
  freqs = [28.2, 38.2]
  collectDayDataDual(sm, COLLECT_PORT0, COLLECT_PORT1, 5, 1000, 1, freqs)
  
  # temp, t = sm.send("T", -1)
  # fileName = t + "F" + str(freq) + "C" + str(collectTime)
  # print(fileName)
  # saveDataDual(sm, "COM21", "COM2", fileName, collectTime, t)
  
  # fileName = "D20260220T185829F38.2C5"
  # fileName = "D20000000T190359F38.2C5"
  # fileName = "D20000000T192656F28.2C5"
  # for fileName in outFiles:
  # readDataDual(fileName, lower = -150, upper = -80, clockFreq=clockFreq, decimation=decimation, PLOT_FREQ = True)
  

if __name__ == '__main__':
  main()