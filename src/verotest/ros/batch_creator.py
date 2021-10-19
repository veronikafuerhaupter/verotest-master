from imagehandler_new import Imagehandler
import os
import re
import numpy as np

imagehandler = Imagehandler()

class Imagecutter:

    batch_list = None
    counter = 0

    def __init__(self):
        #self.batch_list = []
        #self.color_list = []
        #self.depth_list = []
        self.directory_list = []


    def iterate_file(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.directory_list.append('training_material0')
        self.directory_list.append('training_material90')
        self.directory_list.append('training_material180')
        runner = 0

        for i in self.directory_list:
            directory_name = self.directory_list[runner]
            dir_path_training = os.path.join(dir_path, directory_name)
            runner = runner + 1

            for dirpath, dirnames, filenames in sorted(os.walk(dir_path_training)):

                for dirname in dirnames:
                    counter = 0
                    dir_path_observations = os.path.join(dir_path_training, dirname)
                    #self.color_list.clear()
                    #self.depth_list.clear()

                    for dirpath, dirnames, filenames in sorted(os.walk(dir_path_observations)):

                        for filename in filenames:
                            counter = counter + 1

                            if counter > 8:
                                #if len(self.color_list) == 8 and len(self.depth_list) == 8:
                                    #self.batch_list.append({'color': [self.color_list], 'depth': [self.depth_list], 'label': [runner]})
                                break

                            if counter < 9:
                                col_img_path = os.path.join(dir_path_observations, 'img_col' + str(counter) + '.npy')
                                depth_img_path = os.path.join(dir_path_observations, 'img_depth' + str(counter) + '.npy')
                                col_img, depth_img = imagehandler.handle_pairs(col_img_path, depth_img_path)

                                # crop permitted area in order to get only images that can be identified with its alignment
                                color_cropped, depth_cropped, color_width, color_height = imagehandler.crop_perm_area(col_img, depth_img)

                                # predict the pallet from the permitted area via Pallet Model from Azure ML
                                predictions = imagehandler.predict_pallet(color_cropped)

                                if predictions is None:
                                    print('No Pallet detected')
                                    continue

                                # further processing if pallet detected in permitted area
                                elif predictions[0]['probability'] > 0.8:
                                    print('Pallet detected')
                                    pallet_depth_cropped, pallet_color_cropped, pallet_color_height, pallet_color_width = imagehandler.crop_pallet(predictions, color_width, color_height, color_cropped, depth_cropped)

                                    # predict if Springmittel is in pallet via the inner radius
                                    circles = imagehandler.predict_circles(pallet_color_cropped)

                                    if circles is None:
                                        print('No Circles detected')
                                        continue

                                    # crop the Springmittel according to its radius
                                    if circles is not None:
                                        # Crop predicted Springmittel in the pallet
                                        springmittel_color_cropped, springmittel_depth_cropped = imagehandler.crop_circles(circles, pallet_color_cropped, pallet_depth_cropped)
                                        directory_col = dir_path_observations + '/img_col_cropped' + str(counter)
                                        directory_depth = dir_path_observations + '/img_depth_cropped' + str(counter)

                                        np.save(directory_col, springmittel_color_cropped)
                                        np.save(directory_depth, springmittel_depth_cropped)

                                        #self.color_list.append(springmittel_color_cropped)
                                        #self.depth_list.append(springmittel_depth_cropped)
                                        #print('Springmittel detected')

                                    else:
                                        print('No Springmittel detected')
                                        continue

imagecutter = Imagecutter()
imagecutter.iterate_file()


