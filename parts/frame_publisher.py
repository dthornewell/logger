import cv2
import numpy as np
class Frame_Publisher():
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.frame = None
    def run(self):
        ret, self.frame = self.cap.read()   
        cv2.imshow(self.frame)
        return np.array(self.frame)