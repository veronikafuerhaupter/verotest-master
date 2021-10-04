#Average depth values

#Ursprüngliches Bild quadratisch
#Springmittel Maße 56mm x 56mm
#Mittelpunkt 28mm x 28mm, +14mm in beide Richtungen = Mittelwertpixel
import numpy as np
import time
import argparse
import cv2

from imagehandler import Imagehandler
from PIL import Image

class Imagecutter:

    outlierfree_list = None
    final_image_list = None

    def __init__(self):
        self.outlierfree_list = []
        self.final_image_list = []

    def remove_outlier(self, cropped_list):
        """
        Experiments showed that placing a new Springmittel in the tray leads to a high variation in depth data before
        it normalizes again. This variation exceeds the depth difference of 3mm magnificently. Therefore, this method removes
        the outliers (depth changes above 8mm) in order to receive reliable depth difference information.
        """
        length_cropped_list = len(cropped_list)
        counter = 0
        denominator = 0
        numerator = 0

        for index, elem in enumerate(cropped_list):

            # Append first without considering it as an outlier
            if index < 2:
                self.outlierfree_list.append(cropped_list[index])
                im = Image.fromarray(cropped_list[index]['color'])
                im.save("cuttet_cropped" + str(numerator) + ".jpeg")
                continue

            depth_cropped_list = cropped_list[index]['depth']
            ratio_x_depth = depth_cropped_list.shape[1] / 56
            ratio_y_depth = depth_cropped_list.shape[0] / 56

            center_point_x = depth_cropped_list.shape[1] / 2
            center_point_y = depth_cropped_list.shape[0] / 2

            x1 = int(center_point_x - 10 * ratio_x_depth)
            x2 = int(center_point_x + 10 * ratio_x_depth)

            y1 = int(center_point_y - 10 * ratio_y_depth)
            y2 = int(center_point_y + 10 * ratio_y_depth)

            # Consider the inner circle of the Springmittel as surface with changing depth
            avg_depth = np.mean(depth_cropped_list[y1:y2, x1:x2])

            if index == 2:
                self.outlierfree_list.append(cropped_list[index])
                im = Image.fromarray(cropped_list[index]['color'])
                im.save("cuttet_cropped" + str(numerator) + ".jpeg")
                avg = avg_depth
                counter = avg_depth
                denominator = 1
                continue

            # Eliminate depth differences bigger 5mm considering the variation of depth when placing a Springmittel
            elif index > 2:
                x = avg - avg_depth
                if -0.005 < x < 0.005:
                    self.outlierfree_list.append(cropped_list[index])
                    numerator = numerator + 1
                    counter = counter + avg_depth
                    denominator = denominator + 1
                    avg = counter / denominator
                    im = Image.fromarray(cropped_list[index]['color'])
                    im.save("cuttet_cropped"+str(numerator)+".jpeg")
                    continue

        return self.outlierfree_list

    def create_observations(self, outlierfree_list):
        """
        When Placing a new Springmittel in the tray, experiments showed a difference in the depth data.
        The Springmittel have a height difference of 3mm. This difference will be used to detect a new Springmittel.
        When a new one ist placed in the tray, the average depth around the inner circle is approximately 3mm closer to
        the camera. Since the rc_visard 160 has a fault tolerance of 1.5mm the method looks for images that are above
        this value and returns those.
        """
        counter = 0
        length_coutlierfree_list = len(self.outlierfree_list)

        for index, elem in enumerate(self.outlierfree_list):

            #im = Image.fromarray(self.outlierfree_list[index]['color'])
            #im.save("cuttet_cropped"+str(index)+".jpeg")

            # Comparing previous and current depth in order to detect changes in depth when placing a new Springmittel
            if (index+1 <= len(self.outlierfree_list) and index - 1 >= 0):

                #Hough circle detection

                prev_depth = self.outlierfree_list[index - 1]['depth']
                curr_depth = self.outlierfree_list[index]['depth']

                prev_color = self.outlierfree_list[index - 1]['color']
                curr_color = self.outlierfree_list[index]['color']

                ratio_x_prev = prev_depth.shape[1] / 56
                ratio_x_curr = curr_depth.shape[1] / 56

                ratio_y_prev = prev_depth.shape[0] / 56
                ratio_y_curr = curr_depth.shape[0] / 56

                center_point_x_prev = prev_depth.shape[1] / 2
                center_point_x_curr = curr_depth.shape[1] / 2

                center_point_y_prev = prev_depth.shape[0] / 2
                center_point_y_curr = curr_depth.shape[0] / 2

                x1_prev = int(center_point_x_prev - 12 * ratio_x_prev)
                x2_prev = int(center_point_x_prev + 12 * ratio_x_prev)

                x1_curr = int(center_point_x_curr - 12 * ratio_x_curr)
                x2_curr = int(center_point_x_curr + 12 * ratio_x_curr)

                y1_prev = int(center_point_y_prev - 12 * ratio_y_prev)
                y2_prev = int(center_point_y_prev + 12 * ratio_y_prev)

                y1_curr = int(center_point_y_curr - 12 * ratio_y_curr)
                y2_curr = int(center_point_y_curr + 12 * ratio_y_curr)

                avg_depth_prev = np.mean(prev_depth[y1_prev:y2_prev, x1_prev:x2_prev])
                avg_depth_curr = np.mean(curr_depth[y1_curr:y2_curr, x1_curr:x2_curr])

                '''if index == 1:
                    self.final_image_list.append({'observation_color': prev_color})
                    counter = counter + 1
                    im.save('springmittelx' + str(counter) + str(index) + '.jpeg')
                    continue

                if (avg_depth_prev - avg_depth_curr) > 0.0015 and counter == 1:
                    self.final_image_list.append({'observation_color': prev_color})
                    counter = counter + 1
                    im.save('springmittelx' + str(counter) + str(index) + '.jpeg')
                    continue'''

                # If depth difference is above 1.5mm (1.5 mm tolerance of rc_visard 160) new Springmittel is considered, previous safed to a list
                if (avg_depth_prev - avg_depth_curr) > 0.002:
                    if counter < 7:
                        self.final_image_list.append({'observation_color': prev_color})
                        counter = counter + 1
                        im = Image.fromarray(prev_color)
                        im.save('springmittelx' + str(counter) + str(index) + '.jpeg')
                        continue

                # If seven are already saved, the last one from the outlier-free list is automatically added
                elif counter == 7:
                    if index == (length_coutlierfree_list - 1):
                        self.final_image_list.append({'observation_color': curr_color})
                        counter = counter + 1
                        im = Image.fromarray(curr_color)
                        im.save('springmittelx' + str(counter) + str(index) + '.jpeg')
                        if counter == 8:
                            print('The observation has been created successfully')
                            exit(0)

                else:
                    continue

        # Return observation with 8 images
        return self.final_image_list



























