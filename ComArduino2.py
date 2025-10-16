"""
Real-time IQ FFT viewer for Teensy 4.1.

Teensy sends two consecutive binary int16 buffers per frame:
    1) Iout: outLen samples (little-endian int16)
    2) Qout: outLen samples (little-endian int16)

No headers or sentinels. We assume fixed outLen and that I comes first.
"""

import threading
import serial
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# ---------------- USER SETTINGS ----------------
PORT                      = "COM4"   # Windows: "COMx", macOS: "/dev/cu.usbmodem*"
BAUD                      = 3000000        # Teensy HS-USB CDC ignores this but pyserial needs it
FS                        = 4375000          # complex sample rate (Hz)
OUTLEN_COMPLEX            = 8192             # equals outLen on the Teensy (samples per I or Q buffer)
PLOT_TIME_DOMAIN          = False            # True to also plot I/Q vs time
BUFFER_SECS               = 1             # rolling window
FFT_LEN                   = 1024          # power of two; <= buffered samples
# FFT_DB_MIN                = -120             # fixed dB range; set to None for auto
# FFT_DB_MAX                = 0
# ------------------------------------------------

# Derived sizes
BYTES_PER_LANE = OUTLEN_COMPLEX * 2              # int16 bytes for I or Q
BUFFER_SAMPLES = max(int(BUFFER_SECS * FS), FFT_LEN)

iq_ring = deque(maxlen=BUFFER_SAMPLES)           # stores complex float64 samples
stop_flag = {"value": False}

def read_exact(ser: serial.Serial, n: int) -> bytes:
    """Read exactly n bytes (blocking until filled)."""
    buf = bytearray(n)
    mv = memoryview(buf)
    got = 0
    while got < n and not stop_flag["value"]:
        m = ser.readinto(mv[got:])
        if not m:
            continue
        got += m
    return bytes(buf) if got == n else b""


def reader_thread_iq_pair():
    """Read frames as [Iout(outLen int16)][Qout(outLen int16)] → complex array."""
    try:
        with serial.Serial(PORT, BAUD, timeout=1) as ser:
            ser.reset_input_buffer()
            while not stop_flag["value"]:
                # Read I then Q, back-to-back
                i_bytes = read_exact(ser, BYTES_PER_LANE)
                if not i_bytes:
                    continue
                q_bytes = read_exact(ser, BYTES_PER_LANE)
                if not q_bytes:
                    continue
                
                I = np.frombuffer(i_bytes, dtype="<i2").astype(np.float64)
                Q = np.frombuffer(q_bytes, dtype="<i2").astype(np.float64)
                
                # Form complex IQ and append to rolling buffer
                x = I + 1j * Q
                iq_ring.extend(x)
    except serial.SerialException as e:
        print(f"[Serial] {e}")


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
        # fig, ax_f = plt.subplots(1, 1, figsize=(10, 4))
        # fig.suptitle("Live IQ FFT (Teensy 4.1)")
        fig, (ax_f, ax_phase) = plt.subplots(2, 1, figsize=(10, 6))
        fig.suptitle("Live IQ Spectrum (signed)")
        
        ax_t = time_I = time_Q = None
        
    phase_line, = ax_phase.plot([], [], lw=1)
    ax_phase.set_xlabel("Frequency (Hz)")
    ax_phase.set_ylabel("Phase (rad, unwrapped)")
    ax_phase.grid(True)

    freq_line, = ax_f.plot([], [], lw=1)
    ax_f.set_xlabel("Frequency (Hz)")
    ax_f.set_ylabel("Magnitude (dBFS)")
    ax_f.grid(True)
    return fig, ax_t, ax_f, time_I, time_Q, freq_line, phase_line


def update_plots(ax_t, ax_f, time_I, time_Q, freq_line, phase_line, fig, fs, fft_len, window):
    """Update (optional) time-domain and complex FFT."""
    if len(iq_ring) < fft_len:
        return

    data = np.array(iq_ring, dtype=np.complex128)
    x = data[-fft_len:]

    # Time-domain (optional)
    if PLOT_TIME_DOMAIN and ax_t is not None:
        n_show = min(len(data), BUFFER_SAMPLES)
        x_show = data[-n_show:]
        t_axis = np.arange(-n_show, 0) / fs
        time_I.set_data(t_axis, x_show.real)
        time_Q.set_data(t_axis, x_show.imag)
        ax_t.set_xlim(t_axis[0], t_axis[-1])
        if x_show.size:
            y_max = np.max(np.abs([x_show.real, x_show.imag]))
            ax_t.set_ylim(-1.1 * y_max, 1.1 * y_max)

    # Complex FFT, centered (−Fs/2..+Fs/2)
    # xw = (x - np.mean(x)) * window
    X = (np.fft.fft(x, n=fft_len))
    # cg = np.sum(window) / fft_len  # coherent gain
    mag = X / (fft_len * fs)
    mag_db = 20 * np.log10(mag)
    f_axis = np.fft.fftshift(np.fft.fftfreq(fft_len, d=1.0 / (2 * fs)))

    freq_line.set_data(f_axis, np.fft.fftshift(mag_db))
    ax_f.set_xlim(-fs, fs)
    
    phase = np.unwrap(np.angle(X))
    phase_line.set_data(f_axis, phase)
    ax_phase = phase_line.axes
    ax_phase.set_xlim(-fs/2, fs/2)
    # autoscale phase vertically
    ymin, ymax = np.min(phase), np.max(phase)
    pad = 0.1*(ymax - ymin + 1e-6)
    ax_phase.set_ylim(ymin - pad, ymax + pad)
    # if FFT_DB_MIN is not None and FFT_DB_MAX is not None:
    #     ax_f.set_ylim(FFT_DB_MIN, FFT_DB_MAX)
    # else:
    ax_f.set_ylim(np.max(mag_db) - 100, np.max(mag_db) + 5)

    fig.canvas.draw_idle()
    fig.canvas.flush_events()


def main():
    print(f"Opening {PORT} for IQ (I then Q) frames...")
    t = threading.Thread(target=reader_thread_iq_pair, daemon=True)
    t.start()

    fig, ax_t, ax_f, time_I, time_Q, freq_line, phase_line = init_plot()
    window = np.hanning(FFT_LEN).astype(np.float64)

    try:
        print("Running. Close the window or press Ctrl+C to stop.")
        while plt.fignum_exists(fig.number):
            update_plots(ax_t, ax_f, time_I, time_Q, freq_line, phase_line, fig, FS, FFT_LEN, window)
            plt.pause(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        stop_flag["value"] = True
        t.join(timeout=1)
        print("Stopped.")


if __name__ == "__main__":
    main()
