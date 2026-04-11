from serialMessenger import *
from dataIO import *

def main():
  collectTime = 1
  calibrationTime = collectTime / 100
  freq = 38.2
  clockFreq = 95
  decimation = 32
  
  print("Initializing Settings Serial Connection")
  sm = SerialMessenger(port=SETTINGS_PORT, baudrate=9600, timeout=5)
  print("Done Initializing Serial \n")
  
  # sm.send("C", 10, printResult = True)
  sm.send("C", clockFreq, printResult = True)
  
  # sm.send("C", clockFreq, printResult = True)
  sm.send("S", 0, printResult = True)
  sm.send("D", decimation, printResult = True)
  # sm.send("B", 0, printResult = True) # Set output to debug (1)
  
  sm.send("U", collectTime * 1000, printResult = True)
  sm.send("N", 0)
  # sm.send("L", calibrationTime * 1000, printResult = True)
  
  sm.send("G", 0, printResult = True)
  # sm.send("F", freq, printResult = True)

  time.sleep(1)
  print("Ready")
  
  freqs = [38.2, 35.2, 32.2, 29.2]
  # freqs = [38.2]
  
  collectDayDataDual(sm, COLLECT_PORT0, COLLECT_PORT1, 5, 1000, 1, freqs, collectSecond=3)
  
  t = "D20260411T043158"
  
  outFiles = [t + "F" + str(i) + "C" + str(collectTime) for i in freqs]
  
  # saveDataDual(sm, "COM21", "COM2", outFiles[0], collectTime, t)
  
  for fileName in outFiles:
    # readDataDual(fileName, lower = -150, upper = -120, clockFreq=clockFreq, decimation=decimation, PLOT_FREQ = True)
    readDataDualFFT(fileName, clockFreq=clockFreq, decimation=decimation, PLOT_FREQ = True, PLOT_TIME=False, APPLY_WINDOW=False)
    plt.show()
  
  
  # temp, t = sm.send("T", -1)
  
  
  # t = "D20260102T000000"
  # collectTime = 1
  # fileName = t + "F" + str(freq) + "C" + str(collectTime)
  # saveDataDual(sm, "COM21", "COM2", fileName, collectTime, t)
  
  
  # # fileName = "D20260220T185829F38.2C5"
  # # fileName = "D20000000T190359F38.2C5"
  # # fileName = "D20000000T192656F28.2C5"
  # # for fileName in outFiles:
  # # readDataDual(fileName, lower = -110, upper = -100, clockFreq=clockFreq, decimation=decimation, PLOT_FREQ = True)
  # readDataDualFFT(fileName, clockFreq=clockFreq, decimation=decimation, PLOT_FREQ = True, APPLY_WINDOW=False)
  
  
  # t = "D20260102T000001"
  # collectTime = 1
  # fileName = t + "F" + str(freq) + "C" + str(collectTime)
  # saveDataDual(sm, "COM21", "COM2", fileName, collectTime, t)
  
  # readDataDualFFT(fileName, clockFreq=clockFreq, decimation=decimation, PLOT_FREQ = True, APPLY_WINDOW=False)
  
  # plt.show()
  
if __name__ == '__main__':
  main()