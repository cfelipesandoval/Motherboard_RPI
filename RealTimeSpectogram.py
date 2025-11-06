import numpy as np
import serial
import matplotlib.pyplot as plt
import keyboard  # currently unused, but kept

# ========= CONFIG =========
PORT = "COM4"   # e.g., "COM5" on Windows
BAUD = 115200
FS   = 95e6       # sample rate (Hz) of real input
PACKET_N = 8192             # samples PER PACKET (must match Arduino)
SEPARATE_IQ = True          # True: I[0..N-1] then Q[0..N-1]; False: interleaved IQIQ...
NFFT = 2048                 # FFT length (decoupled from PACKET_N)
HOP_SAMPLES = 1          # how many new samples between spectrogram columns
WINDOW = np.hanning
AVG_ALPHA = 1.0
PLOT_DBFS = True
PLOT_TIME_DOMAIN = False
Y_MIN, Y_MAX = -100, 0      # dBFS limits
TIMEOUT_S = 2

# Spectrogram config
SPEC_COLS = 100             # number of time columns in spectrogram
# ==========================



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

# ===== FS/4 MIXER =====
FS4_LUT = np.array([1+0j, -1j, -1+0j, 1j], dtype=np.complex64)

def fs_over_4_mix(i_int16, state):
    """
    FS/4 complex mixing for a stream of real int16 samples.

    i_int16 : 1D np.int16 array
    state   : current LUT index (0..3) for continuity between packets

    returns: complex64 array, new_state (0..3)
    """
    i = i_int16.astype(np.float32) / 32768.0  # normalize to +/-1
    n = np.arange(len(i), dtype=np.int32)
    mix = FS4_LUT[(state + n) & 3]
    x = i * mix
    new_state = (state + len(i)) & 3
    return x.astype(np.complex64), new_state



def main():
    centerFrequency = 30e6
    decimation = 16
    gain = 0
    noise = 1
    

    ser = serial.Serial(PORT, BAUD, timeout=TIMEOUT_S)
    ser.reset_input_buffer()
    
    ser2 = serial.Serial("COM3", 9600, timeout=TIMEOUT_S)
    ser2.reset_input_buffer()

    BYTES_PER_PKT = 2 * PACKET_N  # int16 real samples (FS/4 input)
    rx_bytes = bytearray()

    # FFT prep
    win = WINDOW(NFFT).astype(np.float32)
    cg = win.sum() / NFFT

    # frequency axis (positive freqs only)
    freqs_full = np.fft.fftfreq(NFFT, d=decimation/(2 * FS))
    pos_mask = (freqs_full >= 0) & (freqs_full < 2 * FS / decimation)
    freqs = freqs_full[pos_mask] + centerFrequency - FS / (2 * decimation)
    n_freq = len(freqs)

    # Plot
    plt.ion()

    if PLOT_TIME_DOMAIN:
        fig, (ax_t, ax_f) = plt.subplots(2, 1, figsize=(10, 6))
        fig.suptitle("Live IQ + Spectrogram (FS/4)")
        time_I, = ax_t.plot([], [], lw=1, label="Re(x)")
        time_Q, = ax_t.plot([], [], lw=1, label="Im(x)")
        ax_t.set_xlabel("Time (s)")
        ax_t.set_ylabel("Amplitude")
        ax_t.grid(True)
        ax_t.legend(loc="upper right")
    else:
        fig, ax_f = plt.subplots(1, 1, figsize=(10, 4))
        fig.suptitle("Live Spectrogram (FS/4)")
        ax_t = time_I = time_Q = None

    fig.canvas.manager.set_window_title("Live IQ Spectrogram (no header)")

    # Spectrogram image
    spec_img = None
    spec_data = None  # shape: (n_freq, SPEC_COLS)

    ring = ComplexRing(capacity=max(4*NFFT, 16384))
    ema_mag = None
    samples_since_fft = 0
    fs4_state = 0  # mixer state (0..3) to maintain continuity

    try:
        while plt.fignum_exists(fig.number):
            # fill byte buffer
            chunk = ser.read(ser.in_waiting or 4096)
            if not chunk:
                continue
            rx_bytes.extend(chunk)

            # process full packets
            while len(rx_bytes) >= BYTES_PER_PKT:
                pkt = rx_bytes[:BYTES_PER_PKT]
                del rx_bytes[:BYTES_PER_PKT]

                # Interpret packet as real int16 samples (FS/4 input)
                I = np.frombuffer(memoryview(pkt), dtype='<i2')

                # Apply FS/4 mixing to get complex baseband
                # x, fs4_state = fs_over_4_mix(I, fs4_state)

                # ring.push(x)
                # samples_since_fft += len(x)
                x = I.astype(np.float32) / 32768.0
                ring.push(x)
                samples_since_fft += len(x)

            # hop?
            if samples_since_fft < HOP_SAMPLES:
                continue
            samples_since_fft = 0

            # gather last NFFT samples (zero-pad until filled)
            if ring.count < NFFT:
                tail = ring.tail(ring.count)
                xfft = np.zeros(NFFT, np.complex64)
                xfft[-len(tail):] = tail
            else:
                xfft = ring.tail(NFFT)

            if PLOT_TIME_DOMAIN and ax_t is not None:
                n_show = min(500, len(xfft))
                x_show = xfft[-n_show:]
                t_axis = np.arange(-n_show, 0) / FS

                time_I.set_data(t_axis, x_show.real)
                time_Q.set_data(t_axis, x_show.imag)
                ax_t.set_xlim(t_axis[0], t_axis[-1])
                if x_show.size:
                    y_max = np.max(np.abs([x_show.real, x_show.imag]))
                    ax_t.set_ylim(-1.1 * y_max, 1.1 * y_max)

            # FFT
            X_full = np.fft.fft(xfft * win, n=NFFT) / (NFFT * cg)
            mag = np.abs(X_full)[pos_mask]

            if PLOT_DBFS:
                mag = 20*np.log10(np.maximum(mag, 1e-12))

            # simple exponential averaging in frequency
            ema_mag = mag if ema_mag is None else (AVG_ALPHA*mag + (1-AVG_ALPHA)*ema_mag)

            # Initialize or update spectrogram
            if spec_data is None:
                spec_data = np.full((n_freq, SPEC_COLS), Y_MIN if PLOT_DBFS else 0.0, dtype=np.float32)
                spec_data[:, -1] = ema_mag
                t_span = SPEC_COLS * HOP_SAMPLES / FS  # seconds of history
                spec_img = ax_f.imshow(
                    spec_data,
                    origin='lower',
                    aspect='auto',
                    extent=[-t_span, 0.0, freqs[0], freqs[-1]],
                    vmin=Y_MIN if PLOT_DBFS else None,
                    vmax=Y_MAX if PLOT_DBFS else None
                )
                ax_f.set_xlabel("Time (s, history)")
                ax_f.set_ylabel("Frequency (Hz)")
                ax_f.set_title(
                    f"Spectrogram, NFFT={NFFT}, Fs={FS:,.0f} Hz, hop={HOP_SAMPLES}, "
                    f"RBW≈{FS/NFFT:,.1f} Hz"
                )
                fig.colorbar(spec_img, ax=ax_f, label="Magnitude (dBFS)" if PLOT_DBFS else "Magnitude")
            else:
                # roll left & append new column
                spec_data = np.roll(spec_data, -1, axis=1)
                spec_data[:, -1] = ema_mag
                spec_img.set_data(spec_data)
                # no need to update extent; time spacing is constant

            fig.canvas.draw()
            fig.canvas.flush_events()
            
            if keyboard.is_pressed('up arrow'):
              centerFrequency += 100e3
              freqs_full = np.fft.fftfreq(NFFT, d=decimation/(2 * FS))
              pos_mask = (freqs_full >= 0) & (freqs_full < 2 * FS / decimation)
              freqs = freqs_full[pos_mask] + centerFrequency - FS / (2 * decimation)
              n_freq = len(freqs)
              send = "<F," + str(centerFrequency / 1e6) + ">"
              ser2.write(send.encode())
              print(f"Center frequency: {centerFrequency / 1e6} MHz")
            elif keyboard.is_pressed('down arrow'):
              centerFrequency -= 100e3
              freqs_full = np.fft.fftfreq(NFFT, d=decimation/(2 * FS))
              pos_mask = (freqs_full >= 0) & (freqs_full < 2 * FS / decimation)
              freqs = freqs_full[pos_mask] + centerFrequency - FS / (2 * decimation)
              n_freq = len(freqs)
              send = "<F," + str(centerFrequency / 1e6) + ">"
              ser2.write(send.encode())
              print(f"Center frequency: {centerFrequency / 1e6} MHz")
            elif keyboard.is_pressed('right arrow'):
              if (gain < 100): gain += 1
              send = "<G," + str(gain) + ">"
              ser2.write(send.encode())
              print(f"Gain: {gain}%")
            elif keyboard.is_pressed('left arrow'):
              if (gain > 0): gain -= 1
              send = "<G," + str(gain) + ">"
              ser2.write(send.encode())
              print(f"Gain: {gain}%")
            elif keyboard.is_pressed('n'):
              if(noise): noise = 0
              send = "<N," + str(noise) + ">"
              ser2.write(send.encode())
              print(f"Noise: {bool(noise)}")
              

    except KeyboardInterrupt:
        pass
    finally:
        ser.close()

if __name__ == "__main__":
    main()
