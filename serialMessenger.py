import serial

class SerialMessenger:
    """
    Sends framed messages <M,D> and waits for replies <M,O>.
    M = command letter
    D = data sent
    O = data returned by the microcontroller
    """

    def __init__(self, port, baudrate=115200, timeout=1.0):
        self.ser = serial.Serial(port, baudrate=baudrate, timeout=timeout)

    def _format_message(self, M, D):
        """
        Construct a byte-encoded message <M,D>.
        """
        return f"<{M},{D}>".encode("utf-8")

    def _read_response(self):
        """
        Reads from serial until a full <...> message is received.
        Returns the inner content as a string, e.g. "M,O".
        """
        buffer = ""
        start_found = False

        while True:
            c = self.ser.read(1).decode(errors="ignore")
            if not c:
                raise TimeoutError("Timed out waiting for response.")

            if c == "<":
                start_found = True
                buffer = ""
                continue

            if start_found:
                if c == ">":
                    return buffer  # full message received
                else:
                    buffer += c

    def send(self, M, D, timeout=1):
        """
        Send <M,D> and WAIT for a reply <M_out,O_data>.

        Arguments:
            M: command letter (string)
            D: data payload (anything convertible to string)
            timeout: override serial timeout for this call (seconds)

        Returns:
            (M_out, O_data) as strings
        """

        # Remove stale data in buffer to avoid reading old messages
        self.ser.reset_input_buffer()

        # Temporarily override serial timeout if requested
        original_timeout = self.ser.timeout
        if timeout is not None:
            self.ser.timeout = timeout

        try:
            # Send formatted message
            msg = self._format_message(M, D)
            self.ser.write(msg)

            # Block until reply <M_out,O_data> is received
            response = self._read_response()

            # Parse response: must be "M,O"
            parts = response.split(",")
            if len(parts) != 2:
                raise ValueError(f"Invalid message received: <{response}>")

            M_out, O_data = parts[0], parts[1]
            return M_out, O_data
        except TimeoutError:
          self.ser.write(0x03)
          # Send formatted message
          msg = self._format_message(M, D)
          self.ser.write(msg)

          # Block until reply <M_out,O_data> is received
          response = self._read_response()

          # Parse response: must be "M,O"
          parts = response.split(",")
          if len(parts) != 2:
              raise ValueError(f"Invalid message received: <{response}>")

          M_out, O_data = parts[0], parts[1]
          return M_out, O_data
        
        finally:
            # Restore original timeout
            self.ser.timeout = original_timeout
            
    def close(self):
      self.ser.close()
    def isOpen(self):
      return self.ser.isOpen()
      


from datetime import datetime, timedelta

def add_seconds(utc_str, minutes):
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