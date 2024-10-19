import cv2
import numpy as np
class Both_Publisher():
    def init(self):
        self.cap = cv2.VideoCapture(0)
        self.frame = None
    def run(self):
        ret, self.frame = self.cap.read()
        height, width = self.frame.shape[:2]

        # Split the self.frame in half
        # Assuming we want to split vertically
        left_half = self.frame[:, :width // 2]
        right_half = self.frame[:, width // 2:]
        return [np.array(left_half), np.array(right_half)] 