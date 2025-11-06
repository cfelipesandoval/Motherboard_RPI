######### iq complex but gotta fix things i changed trying to do fs/4

import numpy as np
import serial
import matplotlib.pyplot as plt
import keyboard

# ========= CONFIG =========
PORT = "COM4"   # e.g., "COM5" on Windows
BAUD = 115200
FS   = 125000000 / 1        # complex sample rate (Hz), 1 I+Q pair per complex sample
PACKET_N = 8192         # complex samples PER PACKET (must match Arduino)
SEPARATE_IQ = True      # True: I[0..N-1] then Q[0..N-1]; False: interleaved IQIQ...
NFFT = 1024             # FFT length (decoupled from PACKET_N)
HOP_SAMPLES = 1       # update hop
WINDOW = np.hanning
AVG_ALPHA = 0.3
PLOT_DBFS = True
PLOT_TIME_DOMAIN = False
Y_MIN, Y_MAX = -140, 0
TIMEOUT_S = 2
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

def main():
    ser = serial.Serial(PORT, BAUD, timeout=TIMEOUT_S)
    ser.reset_input_buffer()
    
    # need to implement some sort of test to sweep the frequency range
    # ser1 = serial.Serial("COM3", 9600, timeout=TIMEOUT_S)
    # ser1.reset_input_buffer()

    BYTES_PER_PKT = 4 * PACKET_N  # int16 I + int16 Q
    rx_bytes = bytearray()

    # FFT prep
    win = WINDOW(NFFT).astype(np.float32)
    cg = win.sum() / NFFT
    # freqs = np.fft.fftshift(np.fft.fftfreq(NFFT, d=1.0/FS))
    freqs = np.fft.fftshift(np.fft.fftfreq(NFFT, d=1.0/FS))

    # Plot
    plt.ion()
    
    # currf = 0
    
    if PLOT_TIME_DOMAIN:
        fig, (ax_t, ax_f) = plt.subplots(2, 1, figsize=(10, 6))
        fig.suptitle("Live IQ + FFT (Teensy 4.1)")
        time_I, = ax_t.plot([], [], lw=1, label="I")
        time_Q, = ax_t.plot([], [], lw=1, label="Q")
        ax_t.set_xlabel("Time (s)")
        ax_t.set_ylabel("Amplitude")
        ax_t.grid(True)
        ax_t.legend(loc="upper right")
    else:
        fig, ax_f = plt.subplots(1, 1, figsize=(10, 4))
        fig.suptitle("Live IQ FFT (Teensy 4.1)")
        ax_t = time_I = time_Q = None
        
        
    # fig, ax_f = plt.subplots()
    (line,) = ax_f.plot(freqs, np.zeros_like(freqs))
    ax_f.set_xlim(0, freqs[-1])
    ax_f.set_xlabel("Frequency (Hz)")
    ax_f.set_ylabel("Magnitude (dBFS)" if PLOT_DBFS else "Magnitude")
    if PLOT_DBFS: ax_f.set_ylim(Y_MIN, Y_MAX)
    ax_f.grid(True, which="both", alpha=0.3)
    fig.canvas.manager.set_window_title("Live IQ Spectrum (no header)")

    ring = ComplexRing(capacity=max(4*NFFT, 16384))
    ema_mag = None
    samples_since_fft = 0

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

                if SEPARATE_IQ:
                    # Treat as FS/4 input
                    I = np.frombuffer(memoryview(pkt), dtype='<i2').astype(np.float32)
                    
                    # I = np.frombuffer(memoryview(pkt)[:2*PACKET_N], dtype='<i2').astype(np.float32)
                    # Q = np.frombuffer(memoryview(pkt)[2*PACKET_N:], dtype='<i2').astype(np.float32)
                else:
                    iq16 = np.frombuffer(pkt, dtype='<i2').astype(np.float32)
                    I = iq16[0::2]; Q = iq16[1::2]

                # x = (I + 1j*Q) / 32768.0  # normalize to +/-1 FS
                x = I / 32768.0
                
                ring.push(x)
                samples_since_fft += PACKET_N

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
                # n_show = min(len(data), BUFFER_SAMPLES)
                n_show = 500
                x_show = xfft[-n_show:]
                t_axis = np.arange(-n_show, 0) / FS

                time_I.set_data(t_axis, x_show.real)
                time_Q.set_data(t_axis, x_show.imag)
                ax_t.set_xlim(t_axis[0], t_axis[-1])
                if x_show.size:
                    y_max = np.max(np.abs([x_show.real, x_show.imag]))
                    ax_t.set_ylim(-1.1 * y_max, 1.1 * y_max)
            
            # FFT
            # X = np.fft.fftshift(np.fft.fft(xfft * win, n=NFFT)) / (NFFT * cg)
            X = (np.fft.fft(xfft * win, n=NFFT)) / (NFFT * cg)
            mag = np.abs(X)
            if PLOT_DBFS:
                mag = 20*np.log10(np.maximum(mag, 1e-12))

            ema_mag = mag if ema_mag is None else (AVG_ALPHA*mag + (1-AVG_ALPHA)*ema_mag)
            line.set_ydata(ema_mag)
            ax_f.set_title(f"NFFT={NFFT}, Fs={FS:,.0f} Hz, hop={HOP_SAMPLES}, RBW≈{FS/NFFT:,.1f} Hz")
            fig.canvas.draw(); fig.canvas.flush_events()\
            
        # if keyboard.is_pressed('up arrow'):
            

    except KeyboardInterrupt:
        pass
    finally:
        ser.close()

if __name__ == "__main__":
    main()





##### old


# if PLOT_TIME_DOMAIN and ax_t is not None:
#         # n_show = min(len(data), BUFFER_SAMPLES)
#         n_show = 50
#         x_show = data[-n_show:]
#         t_axis = np.arange(-n_show, 0) / fs
        
        
#         time_I.set_data(t_axis, x_show.real)
#         time_Q.set_data(t_axis, x_show.imag)
#         ax_t.set_xlim(t_axis[0], t_axis[-1])
#         if x_show.size:
#             y_max = np.max(np.abs([x_show.real, x_show.imag]))
#             ax_t.set_ylim(-1.1 * y_max, 1.1 * y_max)


















# """
# Real-time IQ FFT viewer for Teensy 4.1.

# Teensy sends two consecutive binary int16 buffers per frame:
#     1) Iout: outLen samples (little-endian int16)
#     2) Qout: outLen samples (little-endian int16)

# No headers or sentinels. We assume fixed outLen and that I comes first.
# """

# import threading
# import serial
# import numpy as np
# import matplotlib.pyplot as plt
# from collections import deque

# # ---------------- USER SETTINGS ----------------
# PORT                      = "COM4"   # Windows: "COMx", macOS: "/dev/cu.usbmodem*"
# BAUD                      = 3_000_000        # Teensy HS-USB CDC ignores this but pyserial needs it
# FS                        = 120000000 / 16        # complex sample rate (Hz)
# OUTLEN_COMPLEX            = 8192             # equals outLen on the Teensy (samples per I or Q buffer)
# PLOT_TIME_DOMAIN          = False            # True to also plot I/Q vs time
# BUFFER_SECS               = 0.01           # rolling window
# FFT_LEN                   = 256             # power of two; <= buffered samples
# FFT_DB_MIN                = 0             # fixed dB range; set to None for auto
# FFT_DB_MAX                = 150
# # ------------------------------------------------

# # Derived sizes
# BYTES_PER_LANE = OUTLEN_COMPLEX * 2              # int16 bytes for I or Q
# BUFFER_SAMPLES = max(int(BUFFER_SECS * FS), FFT_LEN) * 2

# iq_ring = deque(maxlen=BUFFER_SAMPLES)           # stores complex float64 samples
# start_flag = b"\xAA\x55"
# stop_flag = {"value": False}

# def read_exact(ser: serial.Serial, n: int) -> bytes:
#     """Read exactly n bytes (blocking until filled)."""
#     buf = bytearray(n)
#     mv = memoryview(buf)
#     got = 0
#     while got < n and not stop_flag["value"]:
#         m = ser.readinto(mv[got:])
#         if not m:
#             continue
#         got += m
#     return bytes(buf) if got == n else b""

# def reader_thread_iq_pair():
#     """Read frames as [Iout(outLen int16)][Qout(outLen int16)] → complex array."""
#     try:
#         with serial.Serial(PORT, BAUD, timeout=1) as ser:
#             ser.reset_input_buffer()
#             while not stop_flag["value"]:
#                 # Read I then Q, back-to-back
#                 i_bytes = read_exact(ser, BYTES_PER_LANE)
#                 if not i_bytes:
#                     continue
#                 q_bytes = read_exact(ser, BYTES_PER_LANE)
#                 if not q_bytes:
#                     continue

#                 I = np.frombuffer(i_bytes, dtype="<i2").astype(np.float64)
#                 Q = np.frombuffer(q_bytes, dtype="<i2").astype(np.float64)
                
#                 # with open("Idata.bin", "ab") as f:
#                 #     f.write(I.astype("<i2").tobytes())
#                 # with open("Qdata.bin", "ab") as f:
#                 #     f.write(Q.astype("<i2").tobytes())

#                 # Form complex IQ and append to rolling buffer
#                 x = I + 1j * Q
#                 # x = Q
#                 iq_ring.extend(x)
#     except serial.SerialException as e:
#         print(f"[Serial] {e}")


# def init_plot():
#     """Create figure(s)."""
#     plt.ion()
#     if PLOT_TIME_DOMAIN:
#         fig, (ax_t, ax_f) = plt.subplots(2, 1, figsize=(10, 6))
#         fig.suptitle("Live IQ + FFT (Teensy 4.1)")
#         time_I, = ax_t.plot([], [], lw=1, label="I")
#         time_Q, = ax_t.plot([], [], lw=1, label="Q")
#         ax_t.set_xlabel("Time (s)")
#         ax_t.set_ylabel("Amplitude")
#         ax_t.grid(True)
#         ax_t.legend(loc="upper right")
#     else:
#         fig, ax_f = plt.subplots(1, 1, figsize=(10, 4))
#         fig.suptitle("Live IQ FFT (Teensy 4.1)")
#         ax_t = time_I = time_Q = None

#     freq_line, = ax_f.plot([], [], lw=1)
#     ax_f.set_xlabel("Frequency (Hz)")
#     ax_f.set_ylabel("Magnitude (dBFS)")
#     ax_f.grid(True)
#     return fig, ax_t, ax_f, time_I, time_Q, freq_line

# def update_plots(ax_t, ax_f, time_I, time_Q, freq_line, fig, fs, fft_len, window):
#     """Update (optional) time-domain and complex FFT."""
#     if len(iq_ring) < fft_len:
#         return

#     data = np.array(iq_ring, dtype=np.complex128)
#     x = data[-fft_len:]

#     # Time-domain (optional)
#     if PLOT_TIME_DOMAIN and ax_t is not None:
#         # n_show = min(len(data), BUFFER_SAMPLES)
#         n_show = 50
#         x_show = data[-n_show:]
#         t_axis = np.arange(-n_show, 0) / fs
        
        
#         time_I.set_data(t_axis, x_show.real)
#         time_Q.set_data(t_axis, x_show.imag)
#         ax_t.set_xlim(t_axis[0], t_axis[-1])
#         if x_show.size:
#             y_max = np.max(np.abs([x_show.real, x_show.imag]))
#             ax_t.set_ylim(-1.1 * y_max, 1.1 * y_max)

#     # Complex FFT, centered (−Fs/2..+Fs/2)
#     # xw = (x - np.mean(x)) * window
    
    
#     # X = np.fft.fftshift(np.fft.fft(xw, n=fft_len))
#     # X = np.fft.fftshift(np.fft.fft(x))
#     X = np.fft.fftshift(np.fft.fft(x, n = fft_len))

#     # cg = np.sum(window) / fft_len  # coherent gain
#     mag = np.abs(X) #/ (fft_len * (cg if cg != 0 else 1))
#     mag_db = 20 * np.log10(mag)
#     f_axis = np.fft.fftshift(np.fft.fftfreq(fft_len, d = 1.0 / (4 * fs)))

#     freq_line.set_data(f_axis, mag_db)
#     ax_f.set_xlim(-fs, fs)
#     if FFT_DB_MIN is not None and FFT_DB_MAX is not None:
#         ax_f.set_ylim(FFT_DB_MIN, FFT_DB_MAX)
#     else:
#         ax_f.set_ylim(np.max(mag_db) - 100, np.max(mag_db) + 5)

#     fig.canvas.draw_idle()
#     fig.canvas.flush_events()


# def main():
#     print(f"Opening {PORT} for IQ (I then Q) frames...")
#     t = threading.Thread(target=reader_thread_iq_pair, daemon=True)
#     t.start()

#     fig, ax_t, ax_f, time_I, time_Q, freq_line = init_plot()
#     window = np.hanning(FFT_LEN).astype(np.float64)

#     try:
#         print("Running. `Close` the window or press Ctrl+C to stop.")
#         while plt.fignum_exists(fig.number):
#             update_plots(ax_t, ax_f, time_I, time_Q, freq_line, fig, FS, FFT_LEN, window)
#             plt.pause(0.02)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         stop_flag["value"] = True
#         t.join(timeout=1)
#         print("Stopped.")


# if __name__ == "__main__":
#     main()
