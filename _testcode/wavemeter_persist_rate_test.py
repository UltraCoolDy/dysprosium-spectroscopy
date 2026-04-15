import socket
import time
import csv
import re
import statistics
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# Settings
# -------------------------------------------------------------------
WAVEMETER_HOST = "192.168.20.11"
WAVEMETER_PORT = 7802
WAVEMETER_PORT_NUM = 4
WAVEMETER_CMD = "WAVElength\n"
WAVEMETER_TIMEOUT_S = 0.15

TEST_DURATION_S = 20.0
OUTPUT_CSV = Path("wavemeter_rate_test_persistent_dual_time.csv")

WAVEMETER_THZ_RE = re.compile(r"([-+]?\d+(?:\.\d+)?)\s*THz", re.IGNORECASE)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def recv_one_line(sock: socket.socket, timeout_s: float) -> str:
    sock.settimeout(timeout_s)
    data = b""
    t0 = time.perf_counter()
    while True:
        chunk = sock.recv(4096)
        if not chunk:
            break
        data += chunk
        if b"\n" in data:
            break
        if (time.perf_counter() - t0) > timeout_s:
            break
    return data.decode("ascii", errors="replace").strip()


def wavemeter_set_port(port_num: int, timeout_s: float) -> str:
    with socket.create_connection((WAVEMETER_HOST, WAVEMETER_PORT), timeout=timeout_s) as s:
        s.settimeout(timeout_s)
        cmd = f"OPTSW,SET,{int(port_num)}\n"
        s.sendall(cmd.encode("ascii"))
        return recv_one_line(s, timeout_s)


def read_wavemeter_thz_persistent(sock: socket.socket, timeout_s: float):
    t0_perf = time.perf_counter()
    sock.sendall(WAVEMETER_CMD.encode("ascii"))
    raw = recv_one_line(sock, timeout_s)
    call_dt_s = time.perf_counter() - t0_perf

    m = WAVEMETER_THZ_RE.search(raw)
    thz = float(m.group(1)) if m else None
    return thz, raw, call_dt_s


def summarise(rows):
    ok_rows = [r for r in rows if r["ok"]]
    print("\n===== SUMMARY =====")
    print(f"Total calls:             {len(rows)}")
    print(f"Successful parses:       {len(ok_rows)}")
    print(f"Parse success fraction:  {len(ok_rows) / max(1, len(rows)):.3f}")

    if rows:
        call_dt = [r["call_dt_s"] for r in rows if np.isfinite(r["call_dt_s"])]
        if call_dt:
            print(f"Mean call time (s):      {statistics.mean(call_dt):.6f}")
            print(f"Median call time (s):    {statistics.median(call_dt):.6f}")
            print(f"Max call time (s):       {max(call_dt):.6f}")

    if len(rows) >= 2:
        t_all_pc = [r["pc_time"] for r in rows]
        dt_all_pc = [b - a for a, b in zip(t_all_pc[:-1], t_all_pc[1:])]

        t_all_perf = [r["perf_ns"] for r in rows]
        dt_all_perf = [(b - a) * 1e-9 for a, b in zip(t_all_perf[:-1], t_all_perf[1:])]

        print("\n--- Using pc_time (time.time) ---")
        print(f"Mean dt all calls (s):   {statistics.mean(dt_all_pc):.6f}")
        print(f"Median dt all calls (s): {statistics.median(dt_all_pc):.6f}")
        print(f"Min dt all calls (s):    {min(dt_all_pc):.6f}")
        print(f"Max dt all calls (s):    {max(dt_all_pc):.6f}")
        print(f"Effective call rate:     {1.0 / statistics.mean(dt_all_pc):.2f} Hz")

        print("\n--- Using perf_ns (perf_counter_ns) ---")
        print(f"Mean dt all calls (s):   {statistics.mean(dt_all_perf):.6f}")
        print(f"Median dt all calls (s): {statistics.median(dt_all_perf):.6f}")
        print(f"Min dt all calls (s):    {min(dt_all_perf):.9f}")
        print(f"Max dt all calls (s):    {max(dt_all_perf):.6f}")
        print(f"Effective call rate:     {1.0 / statistics.mean(dt_all_perf):.2f} Hz")

    if len(ok_rows) >= 2:
        t_ok_pc = [r["pc_time"] for r in ok_rows]
        dt_ok_pc = [b - a for a, b in zip(t_ok_pc[:-1], t_ok_pc[1:])]

        t_ok_perf = [r["perf_ns"] for r in ok_rows]
        dt_ok_perf = [(b - a) * 1e-9 for a, b in zip(t_ok_perf[:-1], t_ok_perf[1:])]

        print("\n--- Good samples only: pc_time ---")
        print(f"Mean dt good samp (s):   {statistics.mean(dt_ok_pc):.6f}")
        print(f"Effective good rate:     {1.0 / statistics.mean(dt_ok_pc):.2f} Hz")

        print("\n--- Good samples only: perf_ns ---")
        print(f"Mean dt good samp (s):   {statistics.mean(dt_ok_perf):.6f}")
        print(f"Effective good rate:     {1.0 / statistics.mean(dt_ok_perf):.2f} Hz")


def save_csv(rows, path: Path):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "pc_time",
                "perf_ns",
                "t_rel_pc_s",
                "t_rel_perf_s",
                "call_dt_s",
                "ok",
                "thz",
                "raw",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def make_plots(rows):
    ok_rows = [r for r in rows if r["ok"]]
    if not ok_rows:
        print("No successful samples to plot.")
        return

    t_rel_perf = np.array([r["t_rel_perf_s"] for r in ok_rows], dtype=float)
    thz = np.array([r["thz"] for r in ok_rows], dtype=float)
    perf_ns = np.array([r["perf_ns"] for r in ok_rows], dtype=np.int64)

    dt_perf = np.diff(perf_ns) * 1e-9

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), constrained_layout=True)

    axes[0].plot(t_rel_perf, thz, marker=".", linestyle="-")
    axes[0].set_xlabel("Relative time from perf_counter_ns (s)")
    axes[0].set_ylabel("Frequency (THz)")
    axes[0].set_title("Wavemeter frequency vs time")

    if len(dt_perf) > 0:
        axes[1].plot(np.arange(len(dt_perf)), dt_perf, marker=".", linestyle="-")
        axes[1].set_xlabel("Sample index")
        axes[1].set_ylabel("Δt between good samples (s)")
        axes[1].set_title("Inter-sample spacing from perf_counter_ns")

        axes[2].hist(dt_perf, bins=50)
        axes[2].set_xlabel("Δt between good samples (s)")
        axes[2].set_ylabel("Count")
        axes[2].set_title("Histogram of inter-sample spacing")
    else:
        axes[1].text(0.5, 0.5, "Not enough samples", ha="center", va="center")
        axes[1].set_axis_off()
        axes[2].text(0.5, 0.5, "Not enough samples", ha="center", va="center")
        axes[2].set_axis_off()

    plt.show()


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    print(f"Setting wavemeter port to {WAVEMETER_PORT_NUM}...")
    reply = wavemeter_set_port(WAVEMETER_PORT_NUM, WAVEMETER_TIMEOUT_S)
    print(f"Port set reply: {reply}")

    rows = []

    print(f"Opening persistent socket to {WAVEMETER_HOST}:{WAVEMETER_PORT} ...")
    with socket.create_connection((WAVEMETER_HOST, WAVEMETER_PORT), timeout=WAVEMETER_TIMEOUT_S) as sock:
        sock.settimeout(WAVEMETER_TIMEOUT_S)

        t0_pc = time.time()
        t0_perf_ns = time.perf_counter_ns()
        t_end_pc = t0_pc + TEST_DURATION_S

        print(f"Running persistent-socket max-rate test for {TEST_DURATION_S:.1f} s...")

        i = 0
        while time.time() < t_end_pc:
            pc_time = time.time()
            perf_ns = time.perf_counter_ns()

            t_rel_pc_s = pc_time - t0_pc
            t_rel_perf_s = (perf_ns - t0_perf_ns) * 1e-9

            try:
                thz, raw, call_dt_s = read_wavemeter_thz_persistent(sock, WAVEMETER_TIMEOUT_S)
                ok = thz is not None
            except Exception as e:
                thz = None
                raw = f"ERROR: {e}"
                call_dt_s = np.nan
                ok = False

            rows.append(
                {
                    "index": i,
                    "pc_time": pc_time,
                    "perf_ns": perf_ns,
                    "t_rel_pc_s": t_rel_pc_s,
                    "t_rel_perf_s": t_rel_perf_s,
                    "call_dt_s": call_dt_s,
                    "ok": ok,
                    "thz": thz if ok else "",
                    "raw": raw,
                }
            )
            i += 1

    summarise(rows)
    save_csv(rows, OUTPUT_CSV)
    print(f"Saved CSV: {OUTPUT_CSV}")
    make_plots(rows)


if __name__ == "__main__":
    main()