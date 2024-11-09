import cv2
class Camera():
    def __init__(self) -> None:
        self.img_num = 0
    def run(self):
        if 0 <= self.img_num <= 116:
            self.img_num += 1
        if self.img_num > 116:
            self.img_num = 0
        file_name = "logger/imgs/img_" + str(self.img_num) + ".jpg"
        img = cv2.imread(file_name)
        return img