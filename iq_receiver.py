import serial
import struct


class IQReceiver:
  def __init__(self, port='/dev/ttyACM0', channel_mode=2, timeout=2.0):
    self.ser = serial.Serial(port, baudrate=115200, timeout=timeout)
    self.channel_mode = channel_mode
    self.row_size = 16 if channel_mode == 2 else 8  # bytes per row

  def collect_samples(self, batch_size):
    # Send 'C' command + uint16_t batch size (little endian)
    cmd = b'C' + struct.pack('<H', batch_size)
    self.ser.write(cmd)

  def read_batch(self, expected_rows):
    # Send 'S' to start sending
    self.ser.write(b'S')

    total_bytes = expected_rows * self.row_size
    raw = self.ser.read(total_bytes)

    if len(raw) != total_bytes:
      raise IOError(f"Incomplete read: expected {total_bytes} bytes, got {len(raw)}")

    samples = []
    for i in range(0, len(raw), self.row_size):
      if self.channel_mode == 2:
        iA, qA, tA, iB, qB, tB = struct.unpack('<HHIHHI', raw[i:i + self.row_size])
        samples.append(((iA, qA, tA), (iB, qB, tB)))
      else:
        iA, qA, tA = struct.unpack('<HHI', raw[i:i + self.row_size])
        samples.append((iA, qA, tA))

    return samples

  def close(self):
    self.ser.close()
