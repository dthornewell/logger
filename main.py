
import donkeycar as dk
# import lane_detection_part.lane_detect
from parts.camera import Camera

V = dk.Vehicle()
V.mem['img_num'] = 5
V.add(Camera(), inputs=['img_num'], outputs=['cv_img'])

# V.start(max_loop_count = 5)
V.start(rate_hz = 30)
# while True:
    # cv2.imshow("img", V.mem['cv_img'])



