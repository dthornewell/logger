import pyzed.sl as sl
class Zed_Frame_Publisher():
    def __init__(self):
        # Create a ZED camera object
        self.zed = sl.Camera()

        # Set configuration parameters
        input_type = sl.InputType()
        init = sl.InitParameters(input_t=input_type)
        init.camera_resolution = sl.RESOLUTION.HD1080
        init.depth_mode = sl.DEPTH_MODE.ULTRA
        init.coordinate_units = sl.UNIT.FOOT

        # Open the camera
        err = self.zed.open(init)
        if err != sl.ERROR_CODE.SUCCESS:
            print(repr(err))
            self.zed.close()
            exit(1)

        self.runtime = sl.RuntimeParameters()

        print("ZED camera connected")
    def run(self):
        # Capture a new image
        if self.zed.grab(self.runtime) == sl.ERROR_CODE.SUCCESS:
            # A new image is available if grab() returns ERROR_CODE.SUCCESS
            image = sl.Mat()
            self.zed.retrieve_image(image, sl.VIEW.LEFT)
            imageD = sl.Mat()
            self.zed.retrieve_image(imageD, sl.VIEW.DEPTH)
            return image, imageD
            