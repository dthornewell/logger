import donkeycar as dk
from parts.logger import Logger
from parts.frame_publisher import Frame_Publisher

V = dk.Vehicle()
V.add(Frame_Publisher(), outputs=['left', 'right'])
V.add(Logger(), inputs=['left'])
V.start(rate_hz=30)
