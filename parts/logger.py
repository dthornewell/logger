import datetime
import numpy as np
import cv2
class Logger():
    def __init__(self):
        self.directory = "images_" + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    def run(self, image):
        if image is not None:
            #file_name = self.directory + "\"" + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.f') + ".jpg"
            #cv2.imwrite(file_name, image)
            # print(image)
            cv2.imshow("image", np.array(image))
            #print(file_name)
            cv2.waitKey(1)
        else:
            print("image is None")


