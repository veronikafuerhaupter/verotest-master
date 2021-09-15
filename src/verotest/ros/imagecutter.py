#Average depth values

#Ursprüngliches Bild quadratisch
#Springmittel Maße 57mm x 57mm
#Mittelpunkt 28,5mm x 28,5mm, +14mm in beide Richtungen = Mittelwertpixel
import numpy as np
import time
import argparse
import cv2

from imagehandler import Imagehandler
from PIL import Image

class Imagecutter:

    subtractor_list = None
    final_image_list = None

    def __init__(self):
        self.subtractor_list = []
        self.final_image_list = []

    def create_observations(self, cropped_list):

        length_cropped_list = len(cropped_list)
        counter = 0

        for index, elem in enumerate(cropped_list):
            if (index+1 < length_cropped_list and index - 1 >= 0):

                #Hough circle detection

                prev_depth = cropped_list[index - 1]['depth']
                curr_depth = cropped_list[index]['depth']

                prev_color = cropped_list[index - 1]['color']
                curr_color = cropped_list[index]['color']

                im = Image.fromarray(prev_color)
                im.save("CroppedList"+str(index-1)+".jpg")

                prev_output = prev_color.copy()
                curr_output = curr_color.copy()

                prev_gray = cv2.cvtColor(prev_color, cv2.COLOR_BGR2GRAY)
                curr_gray = cv2.cvtColor(curr_color, cv2.COLOR_BGR2GRAY)


                # detect circles in the image
                prev_circles = cv2.HoughCircles(prev_gray, cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=350, param2=10,
                                           minRadius=10, maxRadius=20)

                curr_circles = cv2.HoughCircles(curr_gray, cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=350, param2=10,
                                           minRadius=10, maxRadius=20)

                if prev_circles is not None:
                    # convert the (x, y) coordinates and radius of the circles to integers
                    circles = np.round(prev_circles[0, :]).astype("int")
                    # loop over the (x, y) coordinates and radius of the circles
                    for (x, y, r) in circles:
                        # draw the circle in the output image, then draw a rectangle
                        # corresponding to the center of the circle
                        cv2.circle(prev_output, (x, y), r, (0, 255, 0), 4)
                        cv2.rectangle(prev_output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
                    # show the output image
                    cv2.imshow("output", np.hstack([prev_color, prev_output]))

                if prev_circles is not None and curr_circles is not None:

                    ratio_prev = ((prev_circles[0][0][2])/8)/2
                    ratio_curr = ((curr_circles[0][0][2])/8)/2

                    center_point_xprev = (prev_circles[0][0][0]) / 2
                    center_point_xcurr = (curr_circles[0][0][0]) / 2

                    center_point_yprev = (prev_circles[0][0][1]) / 2
                    center_point_ycurr = (curr_circles[0][0][1]) / 2

                    x1_prev = int(center_point_xprev - ratio_prev * 6)
                    x2_prev = int(center_point_xprev + ratio_prev * 6)

                    x1_curr = int(center_point_xcurr - ratio_curr * 6)
                    x2_curr = int(center_point_xcurr + ratio_curr * 6)

                    y1_prev = int(center_point_yprev - ratio_prev * 6)
                    y2_prev = int(center_point_yprev + ratio_prev * 6)

                    y1_curr = int(center_point_ycurr - ratio_curr * 6)
                    y2_curr = int(center_point_ycurr + ratio_curr * 6)

                    avg_depth_prev = np.mean(prev_depth[y1_prev:y2_prev, x1_prev:x2_prev])
                    avg_depth_curr = np.mean(curr_depth[y1_curr:y2_curr, x1_curr:x2_curr])

                    if 0.0015 < (avg_depth_prev - avg_depth_curr) < 0.01 and counter < 7:
                        self.subtractor_list.append({'observation_color': prev_color, 'observation_depth': prev_depth})
                        self.final_image_list.append({'observation_color': prev_color})
                        counter = 1
                        im = Image.fromarray(prev_color)
                        im.save('springmittelx'+str(counter)+'.jpeg')
                        counter = counter + 1
                        if counter == 7:
                            x = avg_depth_prev

                    elif counter == 7 and index == length_cropped_list and 0.0015 < (x - avg_depth_curr) < 0.008:
                        self.subtractor_list.append({'observation_color': curr_color, 'observation_depth': curr_depth})
                        self.final_image_list.append({'observation_color': curr_color})
                        counter = counter + 1
                        im = Image.fromarray(curr_color['observation_color'])
                        im.save('springmittelx' + str(index) + '.jpeg')
                        print(self.final_image_list)

                        if counter == 8:
                            print('The observation has been created successfully')
                            exit(0)

        return self.final_image_list

'''ratio_x_prev = prev_depth.shape[1] / 56
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
                               y2_curr = int(center_point_y_curr + 12 * ratio_y_curr)'''
























