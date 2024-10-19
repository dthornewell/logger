import donkeycar.donkeycar as dk
from parts.logger import Logger
from parts.frame_publisher import Frame_Publisher
from lane_detection_part.lane_detect import LaneDetect
V = dk.vehicle.Vehicle()

V.add(Frame_Publisher(), outputs=['frame', 'depth'])
V.add(Logger(), inputs=['frame', 'depth'])
V.add(LaneDetect(), inputs=['frame', 'depth', 'params'])
V.start(rate_hz = 20)
