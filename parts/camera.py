import cv2
class Camera():
    def __init__(self) -> None:
        self.img_num = 0
    def run(self, in_img_num):
        # if 0 <= in_img_num < 116 or in_img_num < 0:
        #     self.img_num = in_img_num
        if 0 <= self.img_num <= 116:
            self.img_num += 1
        if self.img_num > 116:
            self.img_num = 0
        file_name = "logger/imgs/img_" + str(self.img_num) + ".jpg"
        img = cv2.imread(file_name)
        # cv2.imshow("img", img)
        # cv2.waitKey(10)
        # print(img)
        return img
