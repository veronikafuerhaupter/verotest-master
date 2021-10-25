import random

from imagehandler_new import Imagehandler
import os
import re
import numpy as np
from PIL import Image

imagehandler = Imagehandler()

class Batchcreator:

    batch_list = None
    counter = 0

    def __init__(self):
        self.batch_list_col = []
        self.batch_list_depth = []
        self.color_list = []
        self.depth_list = []
        self.directory_list = []

    def iterate_file(self):
        DIR_PATH = os.path.dirname(os.path.realpath(__file__))
        self.directory_list.append('training_material0')
        self.directory_list.append('training_material90')
        self.directory_list.append('training_material180')

        for runner, i in enumerate(self.directory_list):
            directory_name = self.directory_list[runner]
            DIR_PATH_TRAINING = os.path.join(DIR_PATH, directory_name+str('/'))

            #for dirpath, dirnames, filenames in sorted(os.walk(dir_path_training)):
            for f in os.listdir(DIR_PATH_TRAINING):
                DIR_PATH_OBSERVATIONS = os.path.join(DIR_PATH_TRAINING, f+str('/'))
                onlyfiles_col = [f for f in os.listdir(DIR_PATH_OBSERVATIONS) if "img_col_cropped" in f]
                onlyfiles_depth = [f for f in os.listdir(DIR_PATH_OBSERVATIONS) if "img_depth_cropped" in f]
                image_all_col = 0
                image_all_depth = 0

                if len(onlyfiles_col) == 8 and len(onlyfiles_depth) == 8:

                    for counter, file in enumerate(onlyfiles_col):
                        if counter == 0:
                            image = np.load(os.path.join(DIR_PATH_OBSERVATIONS, file))
                            im = Image.fromarray(image)
                            im.save(str(counter) + '.png')
                            image_all_col = np.resize(image, (90, 90, 3))
                        else:
                            image = np.load(os.path.join(DIR_PATH_OBSERVATIONS, file))
                            image_resized = np.resize(image, (90, 90, 3))
                            image_all_col = np.concatenate((image_all_col, image_resized), axis=2)

                    for counter, file in enumerate(onlyfiles_depth):

                        if counter == 0:
                            image = np.load(os.path.join(DIR_PATH_OBSERVATIONS, file))
                            image_all_depth = np.resize(image, (45, 45))
                        else:
                            image = np.load(os.path.join(DIR_PATH_OBSERVATIONS, file))
                            image_resized = np.resize(image, (45, 45))
                            image_all_depth = np.concatenate((image_all_depth, image_resized), axis=1)

                    self.batch_list_col.append({'color': [image_all_col], 'label': [runner]})
                    self.batch_list_depth.append({'depth': [image_all_depth], 'label': [runner]})

        random.shuffle(self.batch_list_col)
        random.shuffle(self.batch_list_depth)

        return self.batch_list_col, self.batch_list_depth


batchcreator = Batchcreator()
batchcreator.iterate_file()