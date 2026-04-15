from __future__ import division

from labscript import *
from labscript_devices.PulseBlasterESRPro200 import PulseBlasterESRPro200 as PulseBlaster
from labscript_devices.NI_DAQmx.models import NI_PXIe_6535

PulseBlaster(name='pulseblaster_0', board_number=0)

ClockLine(
    name='ni6535_0_clock',
    pseudoclock=pulseblaster_0.pseudoclock,
    connection='flag 1'
)

NI_PXIe_6535(
    name='ni_pxi_6535_0',
    parent_device=ni6535_0_clock,
    MAX_name='PXI1Slot4',
    clock_terminal='/PXI1Slot4/PFI4'
)

DigitalOut(
    name='Spec_TTL_pulse',
    parent_device=ni_pxi_6535_0,
    connection='port3/line0'
)

DigitalOut(
    name='Spec_TTL_dummy',
    parent_device=ni_pxi_6535_0,
    connection='port3/line1'
)

print('\n\nCompiling TTL test (NI line, visible pulse)...')

start()

t = 0.5
Spec_TTL_pulse.go_high(t)
Spec_TTL_pulse.go_low(t + 0.1)

t = 0.8
Spec_TTL_pulse.go_high(t)
Spec_TTL_pulse.go_low(t + 0.1)

t = 1.2
stop(t)