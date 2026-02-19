import serial

class SerialMessenger:
    """
    Sends framed messages <M,D> and reads framed replies <M,O>.
    M = command letter
    D = data sent
    O = data returned by the microcontroller
    """

    def __init__(self, port, baudrate=115200, timeout=1.0):
        self.ser = serial.Serial(port, baudrate=baudrate, timeout=timeout)

    def _format_message(self, M, D) -> bytes:
        """Construct a byte-encoded message <M,D>."""
        return f"<{M},{D}>".encode("utf-8")

    def _read_response(self) -> str:
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
                    return buffer
                buffer += c

    # -------------------------
    # New: send-only / recv-only
    # -------------------------

    def send_only(self, M, D, *, flush_input: bool = True) -> None:
        """
        Send <M,D> and return immediately (no waiting for a reply).

        Args:
            M: command letter (string)
            D: data payload (anything convertible to string)
            flush_input: if True, clears stale bytes before sending
        """
        if flush_input:
            self.ser.reset_input_buffer()

        msg = self._format_message(M, D)
        self.ser.write(msg)

    def recv_raw_only(self, *, timeout=None) -> str:
        """
        Receive a framed reply and return the raw inner content, e.g. "M,O".

        Args:
            timeout: temporarily override serial timeout for this call (seconds).
                     Use None to leave current timeout unchanged.

        Returns:
            Raw inner content string (e.g., "M,O")
        """
        original_timeout = self.ser.timeout
        if timeout is not None:
            self.ser.timeout = timeout

        try:
            return self._read_response()
        finally:
            if timeout is not None:
                self.ser.timeout = original_timeout

    def recv_only(self, *, timeout=None):
        """
        Receive a framed reply <M_out,O_data> and parse it.

        Args:
            timeout: temporarily override serial timeout for this call (seconds).
                     Use None to leave current timeout unchanged.

        Returns:
            (M_out, O_data) as strings
        """
        response = self.recv_raw_only(timeout=timeout)

        parts = response.split(",")
        if len(parts) != 2:
            raise ValueError(f"Invalid message received: <{response}>")

        return parts[0], parts[1]

    def send_ctrl_c(self) -> None:
        """Optional utility: send Ctrl-C (ETX) to the device."""
        self.ser.write(b"\x03")

    # -------------------------
    # Optional: keep old behavior as a convenience wrapper
    # -------------------------

    def send(self, M, D, timeout=1, retry = True, printResult = False):
        """
        Convenience method: send then receive.

        If retry=True and a TimeoutError occurs, sends Ctrl-C and tries once more.
        """
        self.send_only(M, D, flush_input=True)

        try:
            val = self.recv_only(timeout=timeout)
            if(printResult): print(val)
            return val
        except TimeoutError:
            if not retry:
                raise
            # self.send_ctrl_c()
            self.send_only(M, D, flush_input=False)
            val = self.recv_only(timeout=timeout)
            if(printResult): print(val)
            return val

    # -------------------------

    def close(self):
        self.ser.close()

    def isOpen(self):
        return self.ser.isOpen()
