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


# Example usage
if __name__ == "__main__":
  esp = ESP32Protocol("/dev/ttyACM0")  # Use your actual USB port here

  try:
    esp.send_command("L", "1")
    print("Sent command to turn on LED")

    esp.send_command("S")
    print("Requested sensor data")

    print("Waiting for response...")
    for _ in range(5):
      response = esp.read_message()
      if response:
        print(f"Received: Command={response['command']}, Data={response['data']}")
        break
      time.sleep(0.5)

  finally:
    esp.close()
