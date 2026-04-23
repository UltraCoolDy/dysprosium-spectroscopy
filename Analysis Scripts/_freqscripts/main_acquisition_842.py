import os
import socket
import time
import csv
import re
import threading
import ctypes
from pathlib import Path

os.add_dll_directory(r"C:\Users\dysprosium\anaconda3\envs\py38\Library\bin")

import cv2
import numpy as np
import pyvisa
import pytesseract
from PIL import ImageGrab
from runmanager.remote import Client

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ============================================================
# USER SETTINGS
# ============================================================
SCOPE_ADDR = "TCPIP0::192.168.20.20::inst0::INSTR"

WAVEMETER_HOST = "192.168.20.11"
WAVEMETER_PORT = 7802
WAVEMETER_SWITCH_PORT = 4 #Wavemeter port labels - 421 = 1, 626 = 3, 842 = 4
WAVEMETER_CMD = "WAVElength\n"
WAVEMETER_TIMEOUT_S = 0.15

OUTPUT_DIR = Path("combined_acq_full")
OUTPUT_DIR.mkdir(exist_ok=True)

SCOPE_TIMEOUT_MS = 20000
POST_SHOT_EXTRA_WAIT_S = 0.5

# OCR temperature grab settings
EC_IMG = (400, 250, 560, 300)
HL_IMG = (1350, 250, 1540, 300)

# Scope trigger settings
TRIG_SOURCE = "CH4"
TRIG_SLOPE = "FALL"
TRIG_LEVEL = 0.282

# ============================================================
# WAVEMETER HELPERS
# ============================================================
WAVEMETER_THZ_RE = re.compile(r"([-+]?\d+(?:\.\d+)?)\s*THz", re.IGNORECASE)


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


def wavemeter_logger(stop_event: threading.Event, rows: list, status: dict):
    try:
        reply = wavemeter_set_port(WAVEMETER_SWITCH_PORT, WAVEMETER_TIMEOUT_S)
        status["port_reply"] = reply

        with socket.create_connection((WAVEMETER_HOST, WAVEMETER_PORT), timeout=WAVEMETER_TIMEOUT_S) as sock:
            sock.settimeout(WAVEMETER_TIMEOUT_S)

            t0_pc = time.time()
            t0_perf_ns = time.perf_counter_ns()

            i = 0
            while not stop_event.is_set():
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

        status["ok"] = True
    except Exception as e:
        status["ok"] = False
        status["error"] = str(e)


def save_wavemeter_csv(rows: list, path: Path):
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

# ============================================================
# OCR TEMPERATURE HELPERS
# ============================================================
user32 = ctypes.windll.user32
SM_XVIRTUALSCREEN = 76
SM_YVIRTUALSCREEN = 77

VX = user32.GetSystemMetrics(SM_XVIRTUALSCREEN)
VY = user32.GetSystemMetrics(SM_YVIRTUALSCREEN)


def img_to_screen(region_img):
    x1, y1, x2, y2 = region_img
    return (x1 + VX, y1 + VY, x2 + VX, y2 + VY)


EC_REGION = img_to_screen(EC_IMG)
HL_REGION = img_to_screen(HL_IMG)


def read_temp(region):
    img = ImageGrab.grab(bbox=region, all_screens=True)
    img = np.array(img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    big = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    config = r'--psm 7 -c tessedit_char_whitelist=0123456789.'
    text = pytesseract.image_to_string(thresh, config=config)

    cleaned = "".join(c for c in text if c.isdigit() or c == ".").strip()

    # Keep only the first sensible number
    m = re.search(r"\d+(?:\.\d+)?", cleaned)
    if not m:
        raise RuntimeError(f"OCR failed to parse temperature from text: {text!r}")

    cleaned = m.group(0)

    return int(round(float(cleaned)))


def get_oven_temps():
    ec = read_temp(EC_REGION)
    hl = read_temp(HL_REGION)
    return ec, hl


# ============================================================
# SCOPE HELPERS
# ============================================================
def read_scope_channel(scope, channel: str):
    scope.write(f"DATA:SOURCE {channel}")
    scope.write("DATA:ENCDG SRIBINARY")
    scope.write("DATA:WIDTH 2")

    ymult = float(scope.query("WFMPRE:YMULT?"))
    yzero = float(scope.query("WFMPRE:YZERO?"))
    yoff = float(scope.query("WFMPRE:YOFF?"))
    xincr = float(scope.query("WFMPRE:XINCR?"))
    xzero = float(scope.query("WFMPRE:XZERO?"))
    ptoff = float(scope.query("WFMPRE:PT_OFF?"))

    record_length = int(scope.query("HORIZONTAL:RECORDLENGTH?"))
    scope.write("DATA:START 1")
    scope.write(f"DATA:STOP {record_length}")

    raw = scope.query_binary_values("CURVE?", datatype="h", container=np.array)
    volts = (raw - yoff) * ymult + yzero
    time_axis = xzero + (np.arange(len(raw)) - ptoff) * xincr

    return {
        "time": time_axis,
        "volts": volts,
        "record_length": record_length,
        "ymult": ymult,
        "yzero": yzero,
        "yoff": yoff,
        "xincr": xincr,
        "xzero": xzero,
        "ptoff": ptoff,
    }


# ============================================================
# MAIN
# ============================================================
def main():
    ec_temp, hl_temp = get_oven_temps()
    run_basename = f"{time.strftime('%Y%m%d_%H%M%S')}_EC{ec_temp}_HL{hl_temp}_full"
    print(f"Oven temps grabbed: EC={ec_temp}, HL={hl_temp}")

    scope_npz = OUTPUT_DIR / f"{run_basename}_scope_all.npz"
    wavemeter_csv = OUTPUT_DIR / f"{run_basename}_wavemeter.csv"

    wm_rows = []
    wm_status = {}
    wm_stop = threading.Event()

    print("Starting wavemeter logger...")
    wm_thread = threading.Thread(
        target=wavemeter_logger,
        args=(wm_stop, wm_rows, wm_status),
        daemon=True,
    )
    wm_thread.start()
    time.sleep(0.5)

    rm = pyvisa.ResourceManager()
    scope = rm.open_resource(SCOPE_ADDR)

    scope.timeout = SCOPE_TIMEOUT_MS
    scope.encoding = "latin_1"
    scope.read_termination = "\n"
    scope.write_termination = "\n"

    print("Connected to scope:", scope.query("*IDN?").strip())

    scope.write("ACQUIRE:STATE STOP")
    scope.write("ACQUIRE:STOPAFTER SEQUENCE")
    scope.write("TRIGGER:A:TYPE EDGE")
    scope.write(f"TRIGGER:A:EDGE:SOURCE {TRIG_SOURCE}")
    scope.write(f"TRIGGER:A:EDGE:SLOPE {TRIG_SLOPE}")
    scope.write(f"TRIGGER:A:LEVEL:{TRIG_SOURCE} {TRIG_LEVEL}")

    scope.write("ACQUIRE:STATE RUN")
    print("Scope armed")

    rmgr = Client()
    rmgr.set_run_shots(True)

    t_shot_launch_pc = time.time()
    t_shot_launch_perf_ns = time.perf_counter_ns()

    print("Triggering labscript shot...")
    rmgr.engage()

    print("Waiting for scope capture to complete...")
    while True:
        state = scope.query("ACQUIRE:STATE?").strip()
        if state == "0":
            break
        time.sleep(0.2)

    print("Scope acquisition complete")
    time.sleep(POST_SHOT_EXTRA_WAIT_S)

    wm_stop.set()
    wm_thread.join(timeout=2.0)
    print("Wavemeter logger stopped")

    time_div = float(scope.query("HORIZONTAL:SCALE?"))
    try:
        sample_rate = float(scope.query("HORIZONTAL:SAMPLERATE?"))
    except Exception:
        sample_rate = np.nan

    ch1 = read_scope_channel(scope, "CH1")
    ch2 = read_scope_channel(scope, "CH2")
    ch3 = read_scope_channel(scope, "CH3")
    ch4 = read_scope_channel(scope, "CH4")

    np.savez(
        scope_npz,
        t_shot_launch_pc=t_shot_launch_pc,
        t_shot_launch_perf_ns=t_shot_launch_perf_ns,
        time_div=time_div,
        sample_rate=sample_rate,
        ec_temp=ec_temp,
        hl_temp=hl_temp,

        time=ch1["time"],

        ch1=ch1["volts"],
        ch2=ch2["volts"],
        ch3=ch3["volts"],
        ch4=ch4["volts"],

        record_length=ch1["record_length"],
        xincr=ch1["xincr"],
        xzero=ch1["xzero"],
        ptoff=ch1["ptoff"],
    )
    print(f"Saved scope data: {scope_npz}")

    save_wavemeter_csv(wm_rows, wavemeter_csv)
    print(f"Saved wavemeter data: {wavemeter_csv}")
    print("Wavemeter status:", wm_status)
    print(f"Wavemeter samples collected: {len(wm_rows)}")

    scope.close()
    rm.close()


if __name__ == "__main__":
    main()