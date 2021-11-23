from imagehandler_new import Imagehandler
import os
import re
import numpy as np
from PIL import Image
from pathlib import Path

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

                                if counter == 1:
                                    # predict the pallet from the permitted area via Pallet Model from Azure ML
                                    predictions = imagehandler.predict_pallet(color_cropped)

                                    if not predictions:
                                        print('No Pallet detected')
                                        continue

                                    # further processing if pallet detected in permitted area
                                    elif predictions[0]['probability'] > 0.8:
                                        print('Pallet detected')

                                        # determine coordinates for cutting the pallet
                                        left, top, right, bottom = imagehandler.determine_coordinates(predictions, color_width, color_height)

                                        # crop pallets with coordinates of first image expecting them not to change in the process of assembling
                                        pallet_depth_cropped, pallet_color_cropped, pallet_color_height, pallet_color_width = imagehandler.crop_pallet(
                                            color_cropped, depth_cropped, left, top, right, bottom)

                                        # identify circles within the Springmittel
                                        circles = imagehandler.predict_circles(pallet_color_cropped)

                                        if circles is None:
                                            print('No Springmittel detected')
                                            break

                                        # crop Springmittel according to the circles coordinates
                                        springmittel_color_cropped, springmittel_depth_cropped = imagehandler.crop_circles(circles, pallet_color_cropped, pallet_depth_cropped)

                                        if springmittel_color_cropped is not None:
                                            # save cropped Springmittel
                                            directory_col = dir_path_observations + '\\springmittel_col_cropped' + str(counter)
                                            depth_cropped = 'springmittel_depth_cropped' + str(counter) + '.npy'
                                            directory_depth = dir_path_observations + '\\springmittel_depth_cropped' + str(counter)
                                            im = Image.fromarray(springmittel_color_cropped)
                                            file_depth = 'springmittel_depth_cropped' + str(counter) + '.npy'
                                            #my_file = 'img_springmittel_col_cropped'+str(counter)+'.png'
                                            #im.save(str(dir_path_observations) + '\\img_springmittel' + str(counter) + '.png')
                                            owd = os.getcwd()
                                            os.chdir(dir_path_observations)
                                            print(os.getcwd())
                                            if os.path.exists('springmittel_col_cropped1.npy'):
                                                np.save(file_depth, springmittel_depth_cropped)
                                                os.chdir(owd)
                                                print(os.getcwd())
                                            #np.save(directory_col, springmittel_color_cropped)
                                            #np.save(directory_depth, springmittel_depth_cropped)
                                            continue

                                        else:
                                            break

                                    else:
                                        print('No Pallet detected')
                                        break

                                else:

                                    # predict the pallet from the permitted area via Pallet Model from Azure ML
                                    pallet_depth_cropped, pallet_color_cropped, pallet_color_height, pallet_color_width = imagehandler.crop_pallet(color_cropped,
                                                                                                                            depth_cropped, left, top, right, bottom)
                                    if circles is None:
                                        print('No Springmittel detected')
                                        break

                                    # crop Springmittel according to the circles coordinates of the first detected circle assuming the pallet does not move while assembly
                                    springmittel_color_cropped, springmittel_depth_cropped = imagehandler.crop_circles(circles, pallet_color_cropped, pallet_depth_cropped)

                                    # save cropped Springmittel
                                    directory_col = dir_path_observations + '\\springmittel_col_cropped' + str(counter)
                                    directory_depth = dir_path_observations + '\\springmittel_depth_cropped' + str(counter)
                                    file_depth = 'springmittel_depth_cropped'+str(counter)
                                    im = Image.fromarray(springmittel_color_cropped)
                                    owd = os.getcwd()
                                    os.chdir(dir_path_observations)
                                    print(os.getcwd())
                                    if os.path.exists('springmittel_col_cropped1.npy'):
                                        np.save(file_depth, springmittel_depth_cropped)
                                        os.chdir(owd)
                                        print(os.getcwd())
                                    #im.save(str(dir_path_observations) + '\\img_springmittel' + str(counter) + '.png')
                                    #np.save(directory_col, springmittel_color_cropped)
                                    #np.save(directory_depth, springmittel_depth_cropped)
                                    continue


imagecutter = Imagecutter()
imagecutter.iterate_file()


