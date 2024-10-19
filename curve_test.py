import donkeycar as dk
from parts.lane_detect import LaneDetect
from parts.zed_frame_publisher import Zed_Frame_Publisher
from parts.camera import Camera
import pickle

V = dk.Vehicle()
V.mem['img_num'] = 5
V.add(Camera(), inputs=['img_num'], outputs=['cv_img'])
V.add(LaneDetect(), inputs=['cv_img', ' ', ' '], outputs=['points'])
#Depth is no longer an image but the meausure which is what you want i think
V.add(Zed_Frame_Publisher(), outputs=['left', 'right', 'depth'])

with open("zed_calibration_params.bin", "rb") as f:
    b = pickle.load(f)
V.mem['zed_calibration_params'] = b

# V.add(LaneDetect(), inputs=['left', 'depth', 'zed_calibration_params'], outputs=['points'])

V.start(rate_hz=2)
