import numpy as np
import cv2
class Zed_Viewer():
    def __init__(self):
        pass
    def run(self, image, zed_calibration_params):
        if image is not None:
            camera_matrix = np.array([[zed_calibration_params["fx"], 0, zed_calibration_params["cx"]],
                                       [0, zed_calibration_params["fy"], zed_calibration_params["cy"]],
                                       [0, 0, 1]])
            
            dist_coeffs = np.array([zed_calibration_params["k1"], zed_calibration_params["k2"], zed_calibration_params["p1"], zed_calibration_params["p2"], zed_calibration_params["k3"]])

            undistort = cv2.undistort(image, camera_matrix, dist_coeffs, newCameraMatrix=camera_matrix)
            cv2.imshow("image", np.array(image))
            cv2.imshow("undistort", np.array(undistort))
            cv2.waitKey(1)
        else:
            print("image is None")     