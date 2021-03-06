# import the necessary packages
import argparse
import cv2
import imutils
import numpy as np
import os
from os.path import dirname, realpath, join

from onnxruntime_predict_pallet import ONNXRuntimeObjectDetectionPallet
from onnxruntime_predict_springmittel import ONNXRuntimeObjectDetectionSpringmittel


__dir = dirname(realpath(__file__))

PALLET_MODEL_FILENAME = join(__dir, 'PalletDetectionModel', 'model.onnx')
PALLET_LABELS_FILENAME = join(__dir, 'PalletDetectionModel', 'labels.txt')

SPRINGMITTEL_MODEL_FILENAME = join(__dir, 'SpringmittelDetectionModel', 'model.onnx')
SPRINGMITTEL_LABELS_FILENAME = join(__dir, 'SpringmittelDetectionModel', 'labels.txt')

dir_path = os.path.dirname(os.path.realpath(__file__))


class Imagehandler:

    match_list = None
    cropped_list = None
    batch_list = None

    def __init__(self):
        self.match_list = []
        self.cropped_list = []
        self.batch_list = []

    def handle_pairs(self, col_img_path, depth_img_path):

        col_img = np.load(col_img_path)
        depth_img = np.load(depth_img_path)

        return col_img, depth_img

    def crop_perm_area(self, col_img, depth_img):
        """
        This method crops the colored images. Later on, the final net will need to identify the position of the
        Springmittel Minus.Therefore, it is necessary to eliminate part of the picture where the position of the
        Springmittel in the pallet cannot be identified anymore.At the same time depth image will be cropped according
        to the color image. The dimensions of the image were taken by eye for reasons of practicality.As soon as it is
        no longer possible for the human eye to recognise how a Springmittel is arranged, the pallet must be shifted.
        """
        perm_width_left = 310
        perm_width_right = 1060
        perm_height_top = 40
        perm_height_bottom = 690

        color = col_img
        depth = depth_img

        color_cropped = color[perm_height_top:perm_height_bottom, perm_width_left:perm_width_right]
        depth_cropped = depth[int(perm_height_top / 2):int(perm_height_bottom / 2),
                        int(perm_width_left / 2):int(perm_width_right / 2)]

        shape_color = color_cropped.shape
        shape_depth = depth_cropped.shape

        color_width = color_cropped.shape[1]
        color_height = color_cropped.shape[0]

        return color_cropped, depth_cropped, color_width, color_height

    def predict_pallet(self, color_cropped):
        """
        This method predicts the pallet from an Azure Custom Vision model
        """
        with open(PALLET_LABELS_FILENAME, 'r') as f:
            labels = [l.strip() for l in f.readlines()]

        od_model = ONNXRuntimeObjectDetectionPallet(PALLET_MODEL_FILENAME, labels)
        predictions = od_model.predict_image(color_cropped)

        return predictions

    def crop_pallet(self, color_cropped, depth_cropped, left, top, right, bottom):
        """
        This method crops the predicted pallet in its dimensions
        """
        pallet_depth_cropped = None
        pallet_color_cropped = None
        pallet_color_height = None
        pallet_color_width = None

        pallet_color_cropped = color_cropped[top:bottom, left:right]
        pallet_depth_cropped = depth_cropped[int(top / 2):int(bottom / 2), int(left / 2):int(right / 2)]

        pallet_color_width = pallet_color_cropped.shape[1]
        pallet_color_height = pallet_color_cropped.shape[0]

        pallet_depth_width = pallet_depth_cropped.shape[1]
        pallet_depth_height = pallet_depth_cropped.shape[0]

        print('Pallet cropped')

        return pallet_depth_cropped, pallet_color_cropped, pallet_color_height, pallet_color_width

    def determine_coordinates(self, predictions, color_width, color_height):
        """
        This method determines the pallets location for cropping
        """
        left = int(predictions[0]['boundingBox']['left'] * color_width)
        top = int(predictions[0]['boundingBox']['top'] * color_height)
        right = int(left + predictions[0]['boundingBox']['width'] * color_width)
        bottom = int(top + predictions[0]['boundingBox']['height'] * color_height)

        return left, top, right, bottom

    def predict_circles(self, springmittel_color_cropped):
        """
        This method uses the Hough Circle Transformation for detecting the inner circle within the Springmittel.
        The maximum radius of 20 pixels ensures that the inner circle is detected, whereas the minimum distance of 300 pixels
        to other circles ensures that only one circle on the cropped pallet is detected.
        """
        gray = cv2.cvtColor(springmittel_color_cropped, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=3, minDist=500, param1=700, param2=40, minRadius=9, maxRadius=13)

        return circles

    def calccrop_coordinates(self, circles, pallet_color_cropped, pallet_depth_cropped, pallet_color_height, pallet_color_width, predictions_springmittel):
        """
        This method uses the coordinates from the Springmittel prediction (Bounding Box coordinates) and the coordinates
        from the circle prediction to calculate new coordinates for cropping a more accurate Springmittel
        """
        for i in range(0, len(predictions_springmittel)):
            left = int(predictions_springmittel[i]['boundingBox']['left'] * pallet_color_width)
            top = int(predictions_springmittel[i]['boundingBox']['top'] * pallet_color_height)
            right = int(left + predictions_springmittel[i]['boundingBox']['width'] * pallet_color_width)
            bottom = int(top + predictions_springmittel[i]['boundingBox']['height'] * pallet_color_height)

            ratio_circles = circles[0][0][2] / 8
            distance_circle_x = circles[0][0][0]
            distance_circle_y = circles[0][0][1]
            dist_total_x = left + distance_circle_x
            dist_total_y = top + distance_circle_y
            left_crop = dist_total_x - 28 * ratio_circles
            right_crop = dist_total_x + 28 * ratio_circles
            top_crop = dist_total_y - 28 * ratio_circles
            bottom_crop = dist_total_y + 28 * ratio_circles

            sm_circle_color_cropped = pallet_color_cropped[int(top_crop):int(bottom_crop), int(left_crop):int(right_crop)]
            sm_circle_depth_cropped = pallet_depth_cropped[int(top_crop / 2):int(bottom_crop / 2), int(left_crop / 2):int(right_crop / 2)]

            return sm_circle_color_cropped, sm_circle_depth_cropped

    def crop_circles(self, circles, pallet_color_cropped, pallet_depth_cropped):
        """
        The Hough Circle Transformation gives information about the radius in pixels.
        As a result the method converts the dimensions of the Springmittel into pixels and
        crops the inner circle around the Springmittel around the hole in the middle
        """
        ratio_circles = circles[0][0][2] / 8
        left = int(circles[0][0][0] - 28 * ratio_circles)
        right = int(circles[0][0][0] + 28 * ratio_circles)
        top = int(circles[0][0][1] - 28 * ratio_circles)
        bottom = int(circles[0][0][1] + 28 * ratio_circles)

        if (left > 0 and right > 0 and top > 0 and bottom > 0):

            springmittel_color_cropped = pallet_color_cropped[top:bottom, left:right]
            springmittel_depth_cropped = pallet_depth_cropped[int(top / 2):int(bottom / 2), int(left / 2):int(right / 2)]

            return springmittel_color_cropped, springmittel_depth_cropped

        else:
            springmittel_color_cropped = None
            springmittel_depth_cropped = None
            print('No circles detected')
            return springmittel_color_cropped, springmittel_depth_cropped

    def predict_springmittel(self, pallet_color_cropped):
        """
        Predict the Springmittel from Azure Custom Vision model
        """
        with open(SPRINGMITTEL_LABELS_FILENAME, 'r') as f:
            labels = [l.strip() for l in f.readlines()]

        od_model = ONNXRuntimeObjectDetectionSpringmittel(SPRINGMITTEL_MODEL_FILENAME, labels)
        predictions_springmittel = od_model.predict_image(pallet_color_cropped)

        return predictions_springmittel

    def crop_springmittel(self, predictions_springmittel, pallet_color_width, pallet_color_height, pallet_color_cropped, pallet_depth_cropped):
        """
        This method crops the predicted pallet in its dimensions
        """
        springmittel_depth_cropped = None
        springmittel_color_cropped = None
        springmittel_color_height = None
        springmittel_color_width = None

        for i in range(0, len(predictions_springmittel)):
            left = int(predictions_springmittel[i]['boundingBox']['left'] * pallet_color_width)
            top = int(predictions_springmittel[i]['boundingBox']['top'] * pallet_color_height)
            right = int(left + predictions_springmittel[i]['boundingBox']['width'] * pallet_color_width)
            bottom = int(top + predictions_springmittel[i]['boundingBox']['height'] * pallet_color_height)

            springmittel_color_cropped = pallet_color_cropped[top:bottom, left:right]
            springmittel_depth_cropped = pallet_depth_cropped[int(top / 2):int(bottom / 2), int(left / 2):int(right / 2)]

            springmittel_color_width = springmittel_color_cropped.shape[1]
            springmittel_color_height = springmittel_color_cropped.shape[0]

            springmittel_depth_width = springmittel_depth_cropped.shape[1]
            springmittel_depth_height = springmittel_depth_cropped.shape[0]

            return springmittel_depth_cropped, springmittel_color_cropped, springmittel_color_height, springmittel_color_width

    def handle_cropped_img(self, sm_circle_color_cropped, sm_circle_depth_cropped):

        self.cropped_list.append({'color': sm_circle_color_cropped, 'depth': sm_circle_depth_cropped})
        print('cropped_list'+str(len(self.cropped_list)))
        #self.cropped_list.append([[springmittel_color_cropped], [springmittel_depth_cropped]])

        return self.cropped_list








