import serial
import struct
import time
import numpy as np

class IQReceiver:
    def __init__(self, port='/dev/ttyACM0', baudrate=115200, batch_size=256, timeout=2):
        self.serial = serial.Serial(port, baudrate=baudrate, timeout=timeout)
        self.batch_size = batch_size
        self.packet_size = batch_size * 12  # 4 + 4x2 bytes

    def request_batch(self):
        self.serial.reset_input_buffer()
        self.serial.write(b'GET\n')
        data = self.serial.read(self.packet_size)

        if data == b'DONE\n':
            print("ESP32 says all data sent.")
            return None

        if len(data) != self.packet_size:
            print(f"Incomplete packet: received {len(data)} bytes")
            return None

        return self.parse(data)

    def parse(self, data):
        dtype = np.dtype([
            ('timestamp', '<u4'),
            ('ia', '<u2'),
            ('qa', '<u2'),
            ('ib', '<u2'),
            ('qb', '<u2'),
        ])
        return np.frombuffer(data, dtype=dtype)

    def collect_all(self):
        result = []
        while True:
            batch = self.request_batch()
            if batch is None:
                break
            result.append(batch)
            print(f"Received {len(batch)} samples")

        if result:
            return np.concatenate(result)
        return np.empty((0,), dtype=[
            ('timestamp', '<u4'),
            ('ia', '<u2'),
            ('qa', '<u2'),
            ('ib', '<u2'),
            ('qb', '<u2'),
        ])