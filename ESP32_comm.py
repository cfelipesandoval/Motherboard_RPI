import serial
import time

class ESP32Protocol:
  def __init__(self, port="/dev/ttyACM0", baud=115200):
    self.serial = serial.Serial(port, baud, timeout=1)
    self.START_MARKER = '<'
    self.END_MARKER = '>'
    self.SEPARATOR = ','
    time.sleep(2)  # Wait for USB CDC to be ready

  def send_command(self, command, data=""):
    """Send a command to the ESP32"""
    message = f"{self.START_MARKER}{command}{self.SEPARATOR}{data}{self.END_MARKER}"
    self.serial.write(message.encode())

  def read_message(self):
    """Read a message from the ESP32"""
    message = ""
    char = self.serial.read().decode(errors='ignore')

    # Wait for the start marker
    while char != self.START_MARKER:
      if not char:
        return None
      char = self.serial.read().decode(errors='ignore')

    # Read until end marker
    char = self.serial.read().decode(errors='ignore')
    while char != self.END_MARKER:
      if not char:
        return None
      message += char
      char = self.serial.read().decode(errors='ignore')

    parts = message.split(self.SEPARATOR)
    command = parts[0]
    data = parts[1] if len(parts) > 1 else ""
    return {"command": command, "data": data}

  def close(self):
    self.serial.close()


def main():
  esp = ESP32Protocol("/dev/ttyACM0")  # Use your actual USB port here

  try:
    print("Send command to Reset ADC")
    esp.send_command("R")
    
    response = esp.read_message()
    if response:
      print(f"Received: Command={response['command']}")
      
    print("Send command to set NCO Freq to 38 MHz")
    esp.send_command("N", "38")
    
    response = esp.read_message()
    if response:
      print(f"Received: Command={response['command']}")
      
    print("Send command to set Bandwidth to 2 MHz")
    esp.send_command("B", "2")
    
    response = esp.read_message()
    if response:
      print(f"Received: Command={response['command']}")

  finally:
    esp.close()

# Example usage
if __name__ == "__main__":
  main()
    

'''
Commands:
  "N": Set NCO Frequency
  "B": Set Bandwidth
  "D": Set Decimation
  "G": Set ADC Gain
  "C": Set Clock Frequency
  "C": Set Sampling Clock Frequency
  "R": Reset
'''
