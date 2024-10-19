import donkeycar.donkeycar as dk
from parts.logger import Logger
from parts.frame_publisher import Frame_Publisher
print(dir(dk))
V = dk.vehicle.Vehicle()
V.add(Frame_Publisher(), outputs=['frame'])
V.add(Logger(), inputs=['frame'])
V.start(rate_hz = 20)
