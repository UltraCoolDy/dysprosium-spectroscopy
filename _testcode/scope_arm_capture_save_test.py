import time
import numpy as np
import pyvisa
from runmanager.remote import Client

SCOPE_ADDR = "TCPIP0::192.168.20.20::inst0::INSTR"

rm = pyvisa.ResourceManager()
scope = rm.open_resource(SCOPE_ADDR)

scope.timeout = 15000
scope.encoding = "latin_1"
scope.read_termination = "\n"
scope.write_termination = "\n"

print("Connected to:", scope.query("*IDN?").strip())

# ---------------------------
# Scope setup
# ---------------------------
scope.write("ACQUIRE:STATE STOP")

# We want to READ CH3
scope.write("DATA:SOURCE CH3")
scope.write("DATA:ENCDG SRIBINARY")
scope.write("DATA:WIDTH 2")

# Single acquisition
scope.write("ACQUIRE:STOPAFTER SEQUENCE")

# Trigger from CH4 scan signal
scope.write("TRIGGER:A:TYPE EDGE")
scope.write("TRIGGER:A:EDGE:SOURCE CH4")
scope.write("TRIGGER:A:EDGE:SLOPE FALL")
scope.write("TRIGGER:A:LEVEL:CH4 0.282")

# Arm the scope
scope.write("ACQUIRE:STATE RUN")
print("Scope armed")

# ---------------------------
# Trigger labscript shot
# ---------------------------
rmgr = Client()
rmgr.set_run_shots(True)

print("Triggering labscript shot...")
t_shot_launch = time.time()
rmgr.engage()

print("Waiting for scope capture to complete...")

# Wait until the single-sequence acquisition is finished
while True:
    state = scope.query("ACQUIRE:STATE?").strip()
    if state == "0":
        break
    time.sleep(0.2)

print("Scope acquisition complete")

# ---------------------------
# Read scaling / timing info
# ---------------------------
ymult = float(scope.query("WFMPRE:YMULT?"))
yzero = float(scope.query("WFMPRE:YZERO?"))
yoff  = float(scope.query("WFMPRE:YOFF?"))

xincr = float(scope.query("WFMPRE:XINCR?"))
xzero = float(scope.query("WFMPRE:XZERO?"))
ptoff = float(scope.query("WFMPRE:PT_OFF?"))

time_div = float(scope.query("HORIZONTAL:SCALE?"))
record_length = int(scope.query("HORIZONTAL:RECORDLENGTH?"))

try:
    sample_rate = float(scope.query("HORIZONTAL:SAMPLERATE?"))
except Exception:
    sample_rate = 1.0 / xincr

scope.write("DATA:START 1")
scope.write(f"DATA:STOP {record_length}")

raw = scope.query_binary_values("CURVE?", datatype="h", container=np.array)

volts = (raw - yoff) * ymult + yzero
time_axis = xzero + (np.arange(len(raw)) - ptoff) * xincr

total_time_from_div = time_div * 10.0
total_time_from_xincr = len(raw) * xincr
total_time_from_rate = len(raw) / sample_rate

np.savez(
    "scope_ch3_ttl_capture.npz",
    time=time_axis,
    volts=volts,
    t_shot_launch=t_shot_launch,
    record_length=record_length,
    ymult=ymult,
    yzero=yzero,
    yoff=yoff,
    xincr=xincr,
    xzero=xzero,
    ptoff=ptoff,
    time_div=time_div,
    sample_rate=sample_rate,
    total_time_from_div=total_time_from_div,
    total_time_from_xincr=total_time_from_xincr,
    total_time_from_rate=total_time_from_rate,
)

print(f"Saved scope_ch3_ttl_capture.npz with {len(raw)} points")
print(f"time/div             = {time_div} s/div")
print(f"sample_rate          = {sample_rate} Sa/s")
print(f"record_length        = {record_length}")
print(f"xincr                = {xincr} s")
print(f"total_time_from_div  = {total_time_from_div} s")
print(f"total_time_from_xincr= {total_time_from_xincr} s")
print(f"total_time_from_rate = {total_time_from_rate} s")

scope.close()
rm.close()