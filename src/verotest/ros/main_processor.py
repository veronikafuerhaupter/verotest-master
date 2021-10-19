from time import sleep

from verotest.ros.ros import Ros
from PIL import Image
import os
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))
print(os.getcwd())
ros = Ros()
os.chdir(dir_path)
os.chdir('training_material0')

print(os.getcwd())

class Datacollector:

    latest_col = None
    latest_depth = None

    def handle_col_img(self, img):
        self.latest_col = img

    def handle_depth_img(self, depth):
        self.latest_depth = depth

    def __init__(self):
        counter = 0
        version = 126

        ros.subscribe_color_imgs(self.handle_col_img)
        ros.subscribe_depth_imgs(self.handle_depth_img)

        directory = 'observation'+str(version)

        if not os.path.exists(directory):
            os.makedirs(directory)

        while True:
            input("please press enter")
            #sleep(2)
            counter = counter + 1

            # Save color image as numpy array to directory
            img_col = self.latest_col['img']
            directory_col = 'observation'+str(version)+'/img_col'+str(counter)
            np.save(directory_col, img_col)

            # Save depth image as numpy array to directory
            img_depth = self.latest_depth['depth']
            directory_depth = 'observation'+str(version)+'/img_depth'+str(counter)
            np.save(directory_depth, img_depth)
            #print(os.getcwd())

            # Save color image as png to directory
            im = Image.fromarray(self.latest_col['img'])
            im.save('observation' + str(version) + '/img' + str(counter) + '.png')
            if counter % 8 == 0:
                version = version + 1
                os.makedirs('observation' + str(version))
                counter = 0


datacollector = Datacollector()



