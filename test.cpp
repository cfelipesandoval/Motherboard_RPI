// serial_u32_logger_overlapped.cpp
//
// High-throughput Windows COM-port logger using OVERLAPPED I/O with multiple
// in-flight reads and a dedicated writer thread.
//
// Captures raw bytes from COMx and writes them byte-for-byte to a binary file.
// Stops after a specified duration or Ctrl+C.
//
// Build (Developer Command Prompt for VS):
//   cl /EHsc /O2 /std:c++17 serial_u32_logger_overlapped.cpp
//
// Run:
//   serial_u32_logger_overlapped.exe COM4 921600 data\run_001\out.bin 10
//
// Args:
//   argv[1] COM port (e.g. COM4)
//   argv[2] baud (kept for completeness; CDC ignores it, but harmless)
//   argv[3] output path (folders auto-created)
//   argv[4] duration seconds (double)
//
// Notes:
// - Writes bytes exactly as received.
// - Overlapped reads keep the pipe full for best possible CDC throughput.
// - No uint32 parsing is required on the host to maximize throughput; treat as raw stream.
//   (If you want to enforce 4-byte alignment, you can add a carry buffer at the writer stage.)

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <thread>
#include <vector>

static std::atomic_bool g_running{true};

static BOOL WINAPI ConsoleHandler(DWORD signal)
{
  if (signal == CTRL_C_EVENT ||
      signal == CTRL_BREAK_EVENT ||
      signal == CTRL_CLOSE_EVENT)
  {
    g_running.store(false, std::memory_order_relaxed);
    return TRUE;
  }
  return FALSE;
}

static double now_seconds_monotonic()
{
  static LARGE_INTEGER freq = []{
    LARGE_INTEGER f{};
    QueryPerformanceFrequency(&f);
    return f;
  }();

  LARGE_INTEGER t{};
  QueryPerformanceCounter(&t);
  return double(t.QuadPart) / double(freq.QuadPart);
}

static HANDLE open_and_configure_serial_overlapped(const char* com_name, DWORD baud)
{
  char device_path[64];
  std::snprintf(device_path, sizeof(device_path), "\\\\.\\%s", com_name);

  HANDLE h = CreateFileA(
    device_path,
    GENERIC_READ,
    0,
    nullptr,
    OPEN_EXISTING,
    FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED,
    nullptr
  );

  if (h == INVALID_HANDLE_VALUE)
  {
    std::fprintf(stderr, "CreateFile failed for %s (GetLastError=%lu)\n", device_path, GetLastError());
    return INVALID_HANDLE_VALUE;
  }

  // Best-effort: configure (CDC often ignores baud, but Windows still allows setting DCB).
  DCB dcb{};
  dcb.DCBlength = sizeof(dcb);

  if (!GetCommState(h, &dcb))
  {
    std::fprintf(stderr, "GetCommState failed (GetLastError=%lu)\n", GetLastError());
    CloseHandle(h);
    return INVALID_HANDLE_VALUE;
  }

  dcb.BaudRate = baud;
  dcb.ByteSize = 8;
  dcb.Parity   = NOPARITY;
  dcb.StopBits = ONESTOPBIT;

  dcb.fOutxCtsFlow = FALSE;
  dcb.fOutxDsrFlow = FALSE;
  dcb.fOutX = FALSE;
  dcb.fInX  = FALSE;
  dcb.fDtrControl = DTR_CONTROL_ENABLE;
  dcb.fRtsControl = RTS_CONTROL_ENABLE;

  if (!SetCommState(h, &dcb))
  {
    std::fprintf(stderr, "SetCommState failed (GetLastError=%lu)\n", GetLastError());
    CloseHandle(h);
    return INVALID_HANDLE_VALUE;
  }

  // We are using OVERLAPPED reads, so timeouts are less central, but keep sane settings.
  COMMTIMEOUTS timeouts{};
  timeouts.ReadIntervalTimeout        = 1;
  timeouts.ReadTotalTimeoutConstant   = 1;
  timeouts.ReadTotalTimeoutMultiplier = 0;
  SetCommTimeouts(h, &timeouts);

  // Increase driver buffers (best-effort)
  SetupComm(h, 1 << 20, 1 << 20);

  // Clear stale RX
  PurgeComm(h, PURGE_RXCLEAR | PURGE_TXCLEAR);

  return h;
}

// Simple producer/consumer queue of byte blocks (move-only vector).
struct ByteBlockQueue
{
  std::mutex m;
  std::condition_variable cv;
  std::deque<std::vector<uint8_t>> q;
  bool closed = false;

  void push(std::vector<uint8_t>&& block)
  {
    {
      std::lock_guard<std::mutex> lk(m);
      if (closed) return;
      q.emplace_back(std::move(block));
    }
    cv.notify_one();
  }

  // returns false when closed and empty
  bool pop(std::vector<uint8_t>& out)
  {
    std::unique_lock<std::mutex> lk(m);
    cv.wait(lk, [&]{ return closed || !q.empty(); });
    if (q.empty())
      return false;
    out = std::move(q.front());
    q.pop_front();
    return true;
  }

  void close()
  {
    {
      std::lock_guard<std::mutex> lk(m);
      closed = true;
    }
    cv.notify_all();
  }
};

struct ReadSlot
{
  OVERLAPPED ov{};
  HANDLE event = nullptr;
  std::vector<uint8_t> buf;
  bool pending = false;

  ReadSlot(size_t buf_size)
  {
    std::memset(&ov, 0, sizeof(ov));
    event = CreateEventA(nullptr, TRUE, FALSE, nullptr);
    ov.hEvent = event;
    buf.resize(buf_size);
  }

  ~ReadSlot()
  {
    if (event) CloseHandle(event);
  }

  ReadSlot(const ReadSlot&) = delete;
  ReadSlot& operator=(const ReadSlot&) = delete;
};

int main(int argc, char** argv)
{
  const char* com_port = "COM4";
  DWORD baud = 115200;
  const char* out_path_cstr = "u32_stream.bin";
  double duration_sec = 10.0;

  if (argc >= 2) com_port = argv[1];
  if (argc >= 3) baud = static_cast<DWORD>(std::strtoul(argv[2], nullptr, 10));
  if (argc >= 4) out_path_cstr = argv[3];
  if (argc >= 5) duration_sec = std::atof(argv[4]);

  SetConsoleCtrlHandler(ConsoleHandler, TRUE);

  // Create output directory if needed
  namespace fs = std::filesystem;
  fs::path out_path_fs(out_path_cstr);
  fs::path out_dir = out_path_fs.parent_path();
  if (!out_dir.empty() && !fs::exists(out_dir))
  {
    std::error_code ec;
    fs::create_directories(out_dir, ec);
    if (ec)
    {
      std::fprintf(stderr, "Failed to create output directory: %s\n", ec.message().c_str());
      return 1;
    }
  }

  HANDLE hSerial = open_and_configure_serial_overlapped(com_port, baud);
  if (hSerial == INVALID_HANDLE_VALUE)
    return 1;

  FILE* fout = std::fopen(out_path_cstr, "wb");
  if (!fout)
  {
    std::fprintf(stderr, "Failed to open output file: %s\n", out_path_cstr);
    CloseHandle(hSerial);
    return 1;
  }

  std::printf("Logging from %s (baud=%lu) -> %s for %.3f s\n",
              com_port, baud, out_path_cstr, duration_sec);
  std::printf("Ctrl+C to stop early.\n");

  // Tuning knobs
  const size_t kReadBufSize = 256 * 1024; // 256 KB per in-flight read
  const int    kNumSlots    = 8;          // number of concurrent reads
  const size_t kMaxQueueBytes = 64ull * 1024 * 1024; // backpressure threshold (64 MB)

  ByteBlockQueue queue;
  std::atomic_uint64_t queued_bytes{0};
  std::atomic_uint64_t total_bytes{0};

  // Writer thread: drains blocks and writes to disk.
  std::thread writer([&]{
    // Large user-space buffer for FILE*
    static std::vector<uint8_t> file_buf(8 * 1024 * 1024);
    std::setvbuf(fout, reinterpret_cast<char*>(file_buf.data()), _IOFBF, file_buf.size());

    std::vector<uint8_t> block;
    while (queue.pop(block))
    {
      if (!block.empty())
      {
        size_t wrote = std::fwrite(block.data(), 1, block.size(), fout);
        if (wrote != block.size())
        {
          std::fprintf(stderr, "Disk write error (wrote %zu of %zu). Stopping.\n", wrote, block.size());
          g_running.store(false, std::memory_order_relaxed);
          break;
        }
        queued_bytes.fetch_sub(block.size(), std::memory_order_relaxed);
      }
      block.clear();
    }
    std::fflush(fout);
  });

  // Prepare read slots + events array
  std::vector<std::unique_ptr<ReadSlot>> slots;
  slots.reserve(kNumSlots);
  std::vector<HANDLE> events;
  events.reserve(kNumSlots);

  for (int i = 0; i < kNumSlots; ++i)
  {
    slots.emplace_back(new ReadSlot(kReadBufSize));
    events.push_back(slots.back()->event);
  }

  auto issue_read = [&](ReadSlot& s) -> bool
  {
    ResetEvent(s.event);
    std::memset(&s.ov, 0, sizeof(s.ov));
    s.ov.hEvent = s.event;

    DWORD got = 0;
    BOOL ok = ReadFile(hSerial, s.buf.data(), (DWORD)s.buf.size(), &got, &s.ov);
    if (ok)
    {
      // Immediate completion is possible; signal event for uniform handling.
      SetEvent(s.event);
      s.pending = false;
      return true;
    }

    DWORD err = GetLastError();
    if (err == ERROR_IO_PENDING)
    {
      s.pending = true;
      return true;
    }

    std::fprintf(stderr, "ReadFile failed (GetLastError=%lu)\n", err);
    return false;
  };

  // Start all reads
  for (auto& sp : slots)
  {
    if (!issue_read(*sp))
    {
      g_running.store(false, std::memory_order_relaxed);
      break;
    }
  }

  const double t0 = now_seconds_monotonic();
  double last_report = t0;

  while (g_running.load(std::memory_order_relaxed))
  {
    double t = now_seconds_monotonic();
    if (duration_sec > 0.0 && (t - t0) >= duration_sec)
      break;

    // Simple backpressure: if queue is too large, pause scheduling by waiting briefly.
    if (queued_bytes.load(std::memory_order_relaxed) > kMaxQueueBytes)
    {
      Sleep(1);
      continue;
    }

    // Wait for any read completion (or timeout to re-check duration)
    DWORD wait_ms = 50;
    DWORD w = WaitForMultipleObjects((DWORD)events.size(), events.data(), FALSE, wait_ms);
    if (w == WAIT_TIMEOUT)
      continue;
    if (w == WAIT_FAILED)
    {
      std::fprintf(stderr, "WaitForMultipleObjects failed (GetLastError=%lu)\n", GetLastError());
      break;
    }

    DWORD idx = w - WAIT_OBJECT_0;
    if (idx >= (DWORD)slots.size())
      continue;

    ReadSlot& s = *slots[idx];

    DWORD bytes_this = 0;
    if (!GetOverlappedResult(hSerial, &s.ov, &bytes_this, FALSE))
    {
      DWORD err = GetLastError();
      // If we are stopping, ignore cancellation noise.
      if (!g_running.load(std::memory_order_relaxed) && (err == ERROR_OPERATION_ABORTED))
        break;

      std::fprintf(stderr, "GetOverlappedResult failed (GetLastError=%lu)\n", err);
      break;
    }

    if (bytes_this > 0)
    {
      total_bytes.fetch_add(bytes_this, std::memory_order_relaxed);

      // Copy into a right-sized block and enqueue for writer thread.
      std::vector<uint8_t> block(bytes_this);
      std::memcpy(block.data(), s.buf.data(), bytes_this);
      queued_bytes.fetch_add(bytes_this, std::memory_order_relaxed);
      queue.push(std::move(block));
    }

    // Re-issue the read on this slot
    if (!issue_read(s))
    {
      g_running.store(false, std::memory_order_relaxed);
      break;
    }

    // Periodic reporting (lightweight)
    if ((t - last_report) >= 1.0)
    {
      double elapsed = t - t0;
      uint64_t tb = total_bytes.load(std::memory_order_relaxed);
      uint64_t qb = queued_bytes.load(std::memory_order_relaxed);
      double mbps = (elapsed > 0.0) ? (double(tb) / (1024.0 * 1024.0) / elapsed) : 0.0;

      std::printf("Elapsed=%.1fs  Total=%.2f MB  Rate=%.2f MB/s  Queue=%.2f MB\n",
                  elapsed,
                  double(tb) / (1024.0 * 1024.0),
                  mbps,
                  double(qb) / (1024.0 * 1024.0));
      last_report = t;
    }
  }

  // Stop I/O and drain
  g_running.store(false, std::memory_order_relaxed);

  // Cancel pending reads (important so the loop cannot hang)
  CancelIoEx(hSerial, nullptr);

  // Close queue and join writer
  queue.close();
  if (writer.joinable())
    writer.join();

  std::fflush(fout);
  std::fclose(fout);
  CloseHandle(hSerial);

  double t1 = now_seconds_monotonic();
  double elapsed = t1 - t0;
  uint64_t tb = total_bytes.load(std::memory_order_relaxed);

  std::printf("Done. Duration=%.3fs  Bytes=%llu  (%.2f MB)  AvgRate=%.2f MB/s\n",
              elapsed,
              (unsigned long long)tb,
              double(tb) / (1024.0 * 1024.0),
              (elapsed > 0.0) ? (double(tb) / (1024.0 * 1024.0) / elapsed) : 0.0);

  return 0;
}
