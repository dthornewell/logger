import donkeycar.donkeycar as dk
from parts.logger import Logger
from parts.frame_publisher import Frame_Publisher
from parts.both_publisher import Both_Publisher

V = dk.Vehicle()
V.add(Both_Publisher(), outputs=['left', 'right'])
V.add(Logger(), inputs=['left'])
V.start(rate_hz=30)