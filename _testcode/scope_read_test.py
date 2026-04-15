import pyvisa
import numpy as np

SCOPE_ADDR = "TCPIP0::192.168.20.20::inst0::INSTR"

rm = pyvisa.ResourceManager()
scope = rm.open_resource(SCOPE_ADDR)

scope.timeout = 10000
scope.encoding = "latin_1"
scope.read_termination = "\n"
scope.write_termination = "\n"

print("Connected to:", scope.query("*IDN?").strip())

# Stop acquisition so the waveform is stable while reading
scope.write("ACQUIRE:STATE STOP")

# Select channel
scope.write("DATA:SOURCE CH3")

# 1-byte binary transfer is simplest
scope.write("DATA:ENCDG RIBINARY")
scope.write("DATA:WIDTH 1")

# Read full record
scope.write("DATA:START 1")
scope.write("DATA:STOP 1000000")

# Get scaling info
ymult = float(scope.query("WFMPRE:YMULT?"))
yzero = float(scope.query("WFMPRE:YZERO?"))
yoff  = float(scope.query("WFMPRE:YOFF?"))

xincr = float(scope.query("WFMPRE:XINCR?"))
xzero = float(scope.query("WFMPRE:XZERO?"))
ptoff = float(scope.query("WFMPRE:PT_OFF?"))

# Read binary waveform
raw = scope.query_binary_values("CURVE?", datatype="b", container=np.array)

# Convert to volts
volts = (raw - yoff) * ymult + yzero

# Convert to time axis
time_axis = xzero + (np.arange(len(raw)) - ptoff) * xincr

print(f"Read {len(raw)} points from CH3")
print(f"First 10 volts: {volts[:10]}")

# Save simple test files
np.savez("scope_ch3_test.npz", time=time_axis, volts=volts)

print("Saved scope_ch3_test.npz")

scope.close()
rm.close()