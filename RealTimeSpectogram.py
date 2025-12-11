import numpy as np
import serial
import matplotlib.pyplot as plt
import keyboard  # currently unused, but kept

# ========= CONFIG =========
PORT = "COM4"   # e.g., "COM5" on Windows
BAUD = 3000000
FS   = 1e6       # sample rate (Hz) of real input
PACKET_N = 8192             # samples PER PACKET (must match Arduino)
NFFT = 8192           # FFT length (decoupled from PACKET_N)
HOP_SAMPLES = 1          # how many new samples between spectrogram columns
WINDOW = np.hanning
AVG_ALPHA = 1.0
PLOT_DBFS = False
Y_MIN, Y_MAX = -120, 0      # dBFS limits
TIMEOUT_S = 2

# Spectrogram config
SPEC_COLS = 100             # number of time columns in spectrogram
# ==========================
BYTES_PER_PKT = 2 * PACKET_N

global SAMPLES
SAMPLES = 0

def check_any_number_key_pressed():
    """Checks if any number key (0-9) is currently pressed."""
    number_keys = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    for key in number_keys:
        if keyboard.is_pressed(key):
            return True, key  # Return True and the pressed key
    return False, None



class ComplexRing:
    def __init__(self, capacity):
        self.buf = np.zeros(capacity, np.complex64)
        self.cap = capacity
        self.w = 0
        self.count = 0
    def push(self, x):
        n = len(x)
        n1 = min(n, self.cap - self.w)
        self.buf[self.w:self.w+n1] = x[:n1]
        n2 = n - n1
        if n2:
            self.buf[0:n2] = x[n1:]
        self.w = (self.w + n) % self.cap
        self.count = min(self.count + n, self.cap)
    def tail(self, n):
        n = min(n, self.count)
        start = (self.w - n) % self.cap
        if start + n <= self.cap:
            return self.buf[start:start+n]
        k = self.cap - start
        return np.concatenate((self.buf[start:], self.buf[:n-k]))

def getTimeData(ser, ring, rx_bytes):
  # fill byte buffer
  chunk = ser.read(ser.in_waiting or PACKET_N)
  # rx_bytes = bytearray()
  # chunk = ser.read(ser.in_waiting) # This was commented out???????
  
  if not chunk:
    return None
  rx_bytes.extend(chunk)

  # process full packets
  while len(rx_bytes) >= BYTES_PER_PKT:
    pkt = rx_bytes[:BYTES_PER_PKT]
    del rx_bytes[:BYTES_PER_PKT]

    # Interpret packet as int16 samples
    # data = np.frombuffer(memoryview(pkt), dtype='<i2')
    data = np.frombuffer(memoryview(pkt), dtype='<i2')

    # x = data.astype(np.float16) / 32768.0
    x = data.astype(np.float32) / 20000
    ring.push(x)

  # gather last NFFT samples (zero-pad until filled)
  # if ring.count < NFFT:
  #   tail = ring.tail(ring.count)
  #   timeData = np.zeros(NFFT, np.complex64)
  #   timeData[-len(tail):] = tail
  # else:
  timeData = ring.tail(NFFT)
  # global SAMPLES
  # SAMPLES += 1
  
  return timeData

def getFFT(timeData):
  # FFT prep
  win = WINDOW(NFFT).astype(np.float32)
  cg = win.sum() / NFFT
  
  # frequency axis (positive freqs only)
  freqs_full = np.fft.fftfreq(NFFT, d=1/(2 * FS))
  pos_mask = (freqs_full >= 0) & (freqs_full < 2 * FS)
  freqs = freqs_full[pos_mask] #+ centerFrequency - FS / (2 * decimation)
  # n_freq = len(freqs)
  
  X_full = np.fft.fft(timeData * win, n=NFFT) #/ (NFFT * cg)
  
  return X_full[pos_mask]

def plotSpectrogram(timeData, PLOT_TIME_DOMAIN = False):
  centerFrequency = 0.3e6
  decimation = 2
  gain = 0
  noise = 1
  
  plt.ion()
  fig, ax_f = plt.subplots(1, 1, figsize=(10, 4))

  fig.canvas.manager.set_window_title("Live IQ Spectrogram (no header)")

  # Spectrogram image
  spec_img = None
  spec_data = None  # shape: (n_freq, SPEC_COLS)
  
  for i in range(0, len(timeData), 4096):    
    X_full = getFFT(timeData[i:i+4096])
    n_freq = len(X_full)
    
    mag = np.abs(X_full)

    if PLOT_DBFS:
      # mag = 20*np.log10(np.maximum(mag, 1e-12))
      mag = 20 * np.log10(mag)
    
    # simple exponential averaging in frequency
    # ema_mag = mag if mag is None else (AVG_ALPHA*mag + (1-AVG_ALPHA)*ema_mag)


    # Initialize or update spectrogram
    if spec_data is None:
      spec_data = np.full((n_freq, SPEC_COLS), Y_MIN if PLOT_DBFS else 0.0, dtype=np.float32)
      spec_data[:, -1] = mag
      # t_span = SPEC_COLS * HOP_SAMPLES / FS  # seconds of history
      spec_img = ax_f.imshow(
          spec_data,
          origin='lower',
          aspect='auto',
          extent=[-SPEC_COLS, 0.0, centerFrequency - FS / (decimation * 2), centerFrequency + FS / (decimation * 2)],
          vmin=Y_MIN if PLOT_DBFS else None,
          vmax=Y_MAX if PLOT_DBFS else 10
      )  #freqs[0], freqs[-1]
      ax_f.set_xlabel("Sample")
      ax_f.set_ylabel("Frequency (Hz)")
      ax_f.set_title(
          f"Spectrogram with Center Freq = {centerFrequency/10**6}MHz, Fs={FS/10**6:,.0f} MHz, FFT Samples = {NFFT}"
      )
      fig.colorbar(spec_img, ax=ax_f, label="Magnitude (dBFS)" if PLOT_DBFS else "Magnitude")
    else:
      # roll left & append new column
      spec_data = np.roll(spec_data, -1, axis=1)
      spec_data[:, -1] = mag
      spec_img.set_data(spec_data)
      # no need to update extent; time spacing is constant

    fig.canvas.draw()
    fig.canvas.flush_events()
    
  while plt.fignum_exists(fig.number):
    continue

def realTimeSpectrogram(PLOT_TIME_DOMAIN = False):
  centerFrequency = 0.062e6
  decimation = 8
  gain = 0
  noise = 1
  
  ser = serial.Serial(PORT, BAUD, timeout = TIMEOUT_S)
  ser.reset_input_buffer()

  # Plot
  plt.ion()

  if PLOT_TIME_DOMAIN:
      fig, (ax_t, ax_f) = plt.subplots(2, 1, figsize=(10, 6))
      fig.suptitle("Time Series Data and Spectrogram Plots")
      time_plot, = ax_t.plot([], [], lw=1, label="x")
      ax_t.set_xlabel("Time (s)")
      ax_t.set_ylabel("Amplitude")
      ax_t.grid(True)
      ax_t.legend(loc="upper right")
  else:
      fig, ax_f = plt.subplots(1, 1, figsize=(10, 4))
      ax_t = time_plot = None

  fig.canvas.manager.set_window_title("Live IQ Spectrogram (no header)")

  # Spectrogram image
  spec_img = None
  spec_data = None  # shape: (n_freq, SPEC_COLS)

  ring = ComplexRing(capacity=max(4*NFFT, PACKET_N))
  rx_bytes = bytearray()
  global SAMPLES
  try:
    while plt.fignum_exists(fig.number):
      
      timeData = getTimeData(ser, ring, rx_bytes)
      
      if timeData is None or len(timeData) < 100:
        print("ERROR")
        continue
      
      SAMPLES = 0
      
      X_full = getFFT(timeData)
      n_freq = len(X_full)
      
      mag = np.abs(X_full)

      if PLOT_DBFS:
        # mag = 20*np.log10(np.maximum(mag, 1e-12))
        mag = 20 * np.log10(mag)
      
      # simple exponential averaging in frequency
      # ema_mag = mag if mag is None else (AVG_ALPHA*mag + (1-AVG_ALPHA)*ema_mag)
      
      if PLOT_TIME_DOMAIN and ax_t is not None:
        n_show = min(500, len(timeData))
        x_show = timeData[-n_show:]
        t_axis = np.arange(-n_show, 0) / FS

        time_plot.set_data(t_axis, x_show.real)
        ax_t.set_xlim(t_axis[0], t_axis[-1])
        if x_show.size:
          # y_max = np.max(np.abs([x_show.real, x_show.imag]))
          y_max = 1.125
          ax_t.set_ylim(-1.1 * y_max, 1.1 * y_max)

      # Initialize or update spectrogram
      if spec_data is None:
        spec_data = np.full((n_freq, SPEC_COLS), Y_MIN if PLOT_DBFS else 0.0, dtype=np.float32)
        spec_data[:, -1] = mag
        # t_span = SPEC_COLS * HOP_SAMPLES / FS  # seconds of history
        
        spec_img = ax_f.imshow(
            spec_data,
            origin='lower',
            aspect='auto',
            extent=[SPEC_COLS, 0.0, centerFrequency - FS / (decimation * 2), centerFrequency + FS / (decimation * 2)],
            vmin=Y_MIN if PLOT_DBFS else 0,
            vmax=Y_MAX if PLOT_DBFS else 10
        )  #freqs[0], freqs[-1]
        ax_f.set_xlabel("Sample")
        ax_f.set_ylabel("Frequency (Hz)")
        ax_f.set_title(
            f"Spectrogram with Center Freq = {centerFrequency/10**6}MHz, Fs={FS/10**6:,.0f} MHz, FFT Samples = {NFFT}"
        )
        fig.colorbar(spec_img, ax=ax_f, label="Magnitude (dBFS)" if PLOT_DBFS else "Magnitude")
      else:
        # roll left & append new column
        spec_data = np.roll(spec_data, -1, axis=1)
        spec_data[:, -1] = mag
        spec_img.set_data(spec_data)
        # no need to update extent; time spacing is constant

      fig.canvas.draw()
      fig.canvas.flush_events()
    
    # pressed, key_pressed = check_any_number_key_pressed()
    # if pressed:
    #   # out = int(key_pressed)
    #   send = "<F," + str(key_pressed) + ">"
    #   ser2.write(send.encode())
    #   print(f"Center frequency: {key_pressed} MHz")
    
    # if keyboard.is_pressed('up arrow'):
    #   centerFrequency += 100e3
    #   freqs_full = np.fft.fftfreq(NFFT, d=decimation/(2 * FS))
    #   pos_mask = (freqs_full >= 0) & (freqs_full < 2 * FS / decimation)
    #   freqs = freqs_full[pos_mask] + centerFrequency - FS / (2 * decimation)
    #   n_freq = len(freqs)
    #   send = "<F," + str(centerFrequency / 1e6) + ">"
    #   ser2.write(send.encode())
    #   print(f"Center frequency: {centerFrequency / 1e6} MHz")
    # elif keyboard.is_pressed('down arrow'):
    #   centerFrequency -= 100e3
    #   freqs_full = np.fft.fftfreq(NFFT, d=decimation/(2 * FS))
    #   pos_mask = (freqs_full >= 0) & (freqs_full < 2 * FS / decimation)
    #   freqs = freqs_full[pos_mask] + centerFrequency - FS / (2 * decimation)
    #   n_freq = len(freqs)
    #   send = "<F," + str(centerFrequency / 1e6) + ">"
    #   ser2.write(send.encode())
    #   print(f"Center frequency: {centerFrequency / 1e6} MHz")
    # elif keyboard.is_pressed('right arrow'):
    #   if (gain < 100): gain += 1
    #   send = "<G," + str(gain) + ">"
    #   ser2.write(send.encode())
    #   print(f"Gain: {gain}%")
    # elif keyboard.is_pressed('left arrow'):
    #   if (gain > 0): gain -= 1
    #   send = "<G," + str(gain) + ">"
    #   ser2.write(send.encode())
    #   print(f"Gain: {gain}%")
    # elif keyboard.is_pressed('n'):
    #   if(noise): noise = 0
    #   else: noise = 1
    #   send = "<N," + str(noise) + ">"
    #   ser2.write(send.encode())
    #   print(f"Noise: {bool(noise)}")
            

  except KeyboardInterrupt:
    pass
  finally:
    ser.close()



if __name__ == "__main__":
  realTimeSpectrogram()
