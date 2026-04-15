from runmanager.remote import Client
import time

rm = Client()

print("Connected to runmanager")

rm.set_run_shots(True)
print("run_shots set True")

rm.engage()
print("Engaged shot")

time.sleep(2)

print("Done")