import donkeycar as dk
from parts.logger import Logger
from parts.frame_publisher import Frame_Publisher
from parts.lane_detect import LaneDetect
import pickle

V = dk.Vehicle()
V.add(Frame_Publisher(), outputs=['left', 'right'], threaded=False)
V.add(Logger(), inputs=['left'], threaded=False)
V.add(LaneDetect(), inputs=['left', ' ', ' '], outputs=['points'], threaded=False)

#Depth is no longer an image but the meausure which is what you want i think
# V.add(Zed_Frame_Publisher(), outputs=['left', 'right', 'depth'])

# with open("zed_calibration_params.bin", "rb") as f:
#     b = pickle.load(f)
# V.mem['zed_calibration_params'] = b

# V.add(LaneDetect(), inputs=['left', 'depth', 'zed_calibration_params'], outputs=['points'])

V.start(rate_hz=5)
