import numpy as np
import matplotlib.pyplot as plt

d = np.load("scope_ch3_ttl_capture.npz")
t = d["time"]
v = d["volts"]

print("Samples:", len(v))
print("Min/Max V:", v.min(), v.max())
print("time/div:", float(d["time_div"]))
print("sample_rate:", float(d["sample_rate"]))
print("xincr:", float(d["xincr"]))
print("total_time_from_div:", float(d["total_time_from_div"]))
print("total_time_from_xincr:", float(d["total_time_from_xincr"]))
print("total_time_from_rate:", float(d["total_time_from_rate"]))

threshold = 1.5
indices = np.where(v > threshold)[0]

if len(indices) == 0:
    print("No TTL pulse found above threshold")
    t_ttl = None
else:
    i0 = indices[0]
    t_ttl = t[i0]
    print(f"TTL detected at t = {t_ttl:.9f} s, V = {v[i0]:.3f} V")

plt.plot(t, v)
plt.axhline(threshold, linestyle="--", label="Threshold")
if t_ttl is not None:
    plt.axvline(t_ttl, linestyle="--", label=f"TTL @ {t_ttl:.6f} s")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.title("CH3 TTL capture")
plt.legend()
plt.tight_layout()
plt.show()