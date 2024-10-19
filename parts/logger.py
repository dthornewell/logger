import datetime
import cv2
import os
class Logger():
    def __init__(self):
        self.directory = "images_" + "/" + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + "/"
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
    def run(self, image):
        if image is not None:
            file_name = self.directory + "/" + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + ".jpg"
            
            print(f"Attempting to save file: {file_name}")
            # Save the image and check if it was successful
            success = cv2.imwrite(file_name, image)
            if success:
                print(f"File saved successfully: {file_name}")
            else:
                print("Failed to save the file. Check if the image is valid.")
            cv2.imshow("image", image)
            cv2.waitKey(1)


