#Average depth values

#Ursprüngliches Bild quadratisch
#Springmittel Maße 57mm x 57mm
#Mittelpunkt 28,5mm x 28,5mm, +14mm in beide Richtungen = Mittelwertpixel
import numpy
import time

from imagehandler import Imagehandler
from PIL import Image

class Imagecutter:

    subtractor_list = None
    final_image_list = None

    def __init__(self):
        self.subtractor_list = []
        self.final_image_list = []

    def create_observations(self, cropped_list):

        length_cropped_list = len(cropped_list[0]['springmittel_depth'])

        for index, elem in enumerate(cropped_list):
            if (index+1 < length_cropped_list and index - 1 >= 0):
                prev_depth = cropped_list[index - 1]['springmittel_depth']
                curr_depth = cropped_list[index]['springmittel_depth']

                prev_color = cropped_list[index - 1]['springmittel_color']
                curr_color = cropped_list[index]['springmittel_color']

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

                avg_depth_prev = numpy.mean(prev_depth[y1_prev:y2_prev, x1_prev:x2_prev])
                avg_depth_curr = numpy.mean(curr_depth[y1_curr:y2_curr, x1_curr:x2_curr])

                counter = 0

                if (avg_depth_prev - avg_depth_curr) > 0.0015 and counter < 8:
                    self.subtractor_list.append([prev_color['observation_color'],
                                                 prev_depth['observation_depth']])
                    self.final_image_list.append(prev_color['observation_color'])
                    counter = counter + 1
                    im = Image.fromarray(prev_color['observation_color'])
                    im.save('springmittel'+str(index)+'.jpeg')
                    return counter

                elif counter == 7 and index == length_cropped_list and (self.subtractor_list[6]['observation_depth'] - curr_depth) > 0.0015:
                    self.subtractor_list.append([curr_color['observation_color'],
                                                 curr_depth['observation_depth']])
                    self.final_image_list.append(curr_color)
                    counter = counter + 1
                    im = Image.fromarray(curr_color['observation_color'])
                    im.save('springmittel' + str(index) + '.jpeg')
                    print(self.final_image_list)

                    if counter == 8:
                        print('The observation has been created successfully')
                        exit(0)

                    return self.final_image_list





















