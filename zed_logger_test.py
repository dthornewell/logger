import donkeycar as dk
from parts.logger import Logger
from parts.zed_frame_publisher import Zed_Frame_Publisher
import pickle

V = dk.Vehicle()
V.add(Zed_Frame_Publisher(), outputs=['left', 'right', 'depth'])
V.add(Logger(), inputs=['depth'])

with open("zed_calibration_params.bin", "rb") as f:
    b = pickle.load(f)

V.mem['zed_calibration_params'] = b

print(V.mem['zed_calibration_params'])
V.start(rate_hz=30)
