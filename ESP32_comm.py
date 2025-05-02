import serial
import time

class ESP32Protocol:
  def __init__(self, port="/dev/ttyS0", baud = 115200):
    self.serial = serial.Serial(port, baud, timeout=1)
    self.START_MARKER = '<'
    self.END_MARKER = '>'
    self.SEPARATOR = ','
    time.sleep(2)  # Allow serial connection to establish
  
  def send_command(self, command, data=""):
    """Send a command to the ESP32"""
    message = f"{self.START_MARKER}{command}{self.SEPARATOR}{data}{self.END_MARKER}"
    self.serial.write(message.encode())
  
  def read_message(self):
    """Read a message from the ESP32"""
    message = ""
    char = self.serial.read().decode()
    
    # Wait for the start marker
    while char != self.START_MARKER:
        char = self.serial.read().decode()
        if not char:
            return None
    
    # Read until end marker
    char = self.serial.read().decode()
    while char != self.END_MARKER:
        if char:
            message += char
        char = self.serial.read().decode()
        if not char:
            return None
    
    # Parse the command and data
    parts = message.split(self.SEPARATOR)
    command = parts[0]
    data = parts[1] if len(parts) > 1 else ""
    
    return {"command": command, "data": data}
  
  def close(self):
    """Close the serial connection"""
    self.serial.close()

# Example usage
if __name__ == "__main__":
    esp = ESP32Protocol()
    
    try:
        # Turn on LED
        esp.send_command("L", "1")
        print("Sent command to turn on LED")
        
        # Request sensor data
        esp.send_command("S")
        print("Requested sensor data")
        
        # Wait for response
        print("Waiting for response...")
        for _ in range(5):  # Try to read a few times
            response = esp.read_message()
            if response:
                print(f"Received: Command={response['command']}, Data={response['data']}")
                break
            time.sleep(0.5)
        
    finally:
        esp.close()
