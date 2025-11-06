import threading
import serial
import numpy as np
import matplotlib.pyplot as plt
from collections import deque




# ---------------- USER SETTINGS ----------------
PORT                      = "COM4"   # Windows: "COMx", macOS: "/dev/cu.usbmodem*"
BAUD                      = 3_000_000        # Teensy HS-USB CDC ignores this but pyserial needs it
FS                        = 120000000 / 32        # complex sample rate (Hz)
OUTLEN_COMPLEX            = 8192             # equals outLen on the Teensy (samples per I or Q buffer)
PLOT_TIME_DOMAIN          = True            # True to also plot I/Q vs time
BUFFER_SECS               = 1           # rolling window
FFT_LEN                   = 1024             # power of two; <= buffered samples
FFT_DB_MIN                = 0             # fixed dB range; set to None for auto
FFT_DB_MAX                = 150
# ------------------------------------------------



def init_plot():
    """Create figure(s)."""
    plt.ion()
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

    freq_line, = ax_f.plot([], [], lw=1)
    ax_f.set_xlabel("Frequency (Hz)")
    ax_f.set_ylabel("Magnitude (dBFS)")
    ax_f.grid(True)
    return fig, ax_t, ax_f, time_I, time_Q, freq_line
  
def update_plots(ax_t, ax_f, time_I, time_Q, freq_line, fig, fs, fft_len, window):
    iq_ring = I_ups + 1j * Q_ups
    data = np.array(iq_ring, dtype=np.complex128)
    x = data[-fft_len:]

    # Time-domain (optional)
    if PLOT_TIME_DOMAIN and ax_t is not None:
        # n_show = min(len(data), BUFFER_SAMPLES)
        n_show = 50
        x_show = data[-n_show:]
        t_axis = np.arange(-n_show, 0) / fs
        time_I.set_data(t_axis, x_show.real)
        time_Q.set_data(t_axis, x_show.imag)
        ax_t.set_xlim(t_axis[0], t_axis[-1])
        if x_show.size:
            y_max = np.max(np.abs([x_show.real, x_show.imag]))
            ax_t.set_ylim(-1.1 * y_max, 1.1 * y_max)

    # Complex FFT, centered (−Fs/2..+Fs/2)
    xw = (x - np.mean(x)) * window
    X = np.fft.fftshift(np.fft.fft(xw, n=fft_len))
    cg = np.sum(window) / fft_len  # coherent gain
    mag = np.abs(X) / (fft_len * (cg if cg != 0 else 1))
    mag_db = 20 * np.log10(mag)
    f_axis = np.fft.fftshift(np.fft.fftfreq(fft_len, d=1.0 / (fs)))

    freq_line.set_data(f_axis, mag_db)
    ax_f.set_xlim(-fs, fs)
    if FFT_DB_MIN is not None and FFT_DB_MAX is not None:
        ax_f.set_ylim(FFT_DB_MIN, FFT_DB_MAX)
    else:
        ax_f.set_ylim(np.max(mag_db) - 100, np.max(mag_db) + 5)

    fig.canvas.draw_idle()
    fig.canvas.flush_events()


def main():
  I = np.fromfile("Idata.bin", dtype="<i2")
  Q = np.fromfile("Qdata.bin", dtype="<i2")

  upsample = 4

  for i in I[0:10000]:
    I_ups.append(i)
    for j in range(0,upsample):
      I_ups.append(0)
      
  for q in Q[0:10000]:
    Q_ups.append(q)
    for j in range(0,upsample):
      Q_ups.append(0)


  np.savetxt('I_ups.csv', np.array(I_ups), delimiter=',')
  np.savetxt('Q_ups.csv', np.array(Q_ups), delimiter=',')
  
  val = 500
  # global I_ups, Q_ups
  I_ups = np.loadtxt('I_ups.csv', delimiter=',')
  Q_ups = np.loadtxt('Q_ups.csv', delimiter=',')
  print(f"Opening {PORT} for IQ (I then Q) frames...")

  fig, ax_t, ax_f, time_I, time_Q, freq_line = init_plot()
  window = np.hanning(FFT_LEN).astype(np.float64)

  print("Running. Close the window or press Ctrl+C to stop.")
  while plt.fignum_exists(fig.number):
      update_plots(ax_t, ax_f, time_I, time_Q, freq_line, fig, FS, FFT_LEN, window)
      plt.pause(0.02)



if __name__ == "__main__":
    main()