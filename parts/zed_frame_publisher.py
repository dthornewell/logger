import pyzed.sl as sl
import cv2
import numpy as np
import pickle

class Zed_Frame_Publisher:
    def __init__(self):
        self.zed = sl.Camera()
        init = sl.InitParameters()
        init.camera_resolution = sl.RESOLUTION.HD1080
        init.depth_mode = sl.DEPTH_MODE.ULTRA
        init.coordinate_units = sl.UNIT.METER
        init.depth_minimum_distance = 0.5

        err = self.zed.open(init)
        if err != sl.ERROR_CODE.SUCCESS:
            print(repr(err))
            self.zed.close()
            exit(1)

        self.runtime = sl.RuntimeParameters()

        calibration_params = self.zed.get_camera_information().camera_configuration.calibration_parameters

        # Access intrinsic parameters
        fx = calibration_params.left_cam.fx  # Focal length in x
        fy = calibration_params.left_cam.fy  # Focal length in y
        cx = calibration_params.left_cam.cx  # Principal point x
        cy = calibration_params.left_cam.cy  # Principal point y

        # Turn into a dictionary
        calibration_params = {
            "f_x": fx,
            "f_y": fy,
            "c_x": cx,
            "c_y": cy
        }

        # Dump to binary file
        with open("zed_calibration_params.bin", "wb") as f:
            pickle.dump(calibration_params, f)

        print("ZED camera connected")

    def run(self):
        if self.zed.grab(self.runtime) == sl.ERROR_CODE.SUCCESS:
            left = sl.Mat()
            self.zed.retrieve_image(left, sl.VIEW.LEFT)
            right = sl.Mat()
            self.zed.retrieve_image(right, sl.VIEW.RIGHT)
            depth = sl.Mat()
            # self.zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
            self.zed.retrieve_image(depth, sl.VIEW.DEPTH)
            # cv2.imshow("ZED", image.get_data())
            return np.array(left.get_data()), np.array(right.get_data()), np.array(depth.get_data())
        else:
            print("ERROR")
            return None, None