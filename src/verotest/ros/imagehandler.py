from os.path import dirname, realpath, join

import rospy
import cv2
import numpy as np

from verotest.logger.logger import Logger
from time import time
from typing import List, Tuple, Dict
from PIL import Image

from onnxruntime_predict_pallet import ONNXRuntimeObjectDetectionPallet
from onnxruntime_predict_springmittel import ONNXRuntimeObjectDetectionSpringmittel
from verotest.ros.ros import Ros

__dir = dirname(realpath(__file__))

PALLET_MODEL_FILENAME = join(__dir, '..', 'PalletDetectionModel', 'model.onnx')
PALLET_LABELS_FILENAME = join(__dir, '..', 'PalletDetectionModel', 'labels.txt')

SPRINGMITTEL_MODEL_FILENAME = join(__dir, '..', 'SpringmittelDetectionModel', 'model.onnx')
SPRINGMITTEL_LABELS_FILENAME = join(__dir, '..', 'SpringmittelDetectionModel', 'labels.txt')


class Imagehandler:

    search_list = None
    reference_entry = None
    match_list = None
    cropped_list = None

    def __init__(self):
        self.search_list = []
        self.reference_entry = []
        self.match_list = []
        self.cropped_list = []

    def handle_color_img(self, img):
        """
        This methods receives subscribed images from Ros color topic and saves them as a list.
        """
        self.search_list.append(img)

    def handle_depth_img(self, img):
        """"
        This method receives depth images from Ros depth topic and saves them as as list.
        It executes the find entry method and creates a list with timely matching color and depth information.
        """
        counter = 0
        self.reference_entry.append(img)
        found = self.find_entry_temporal_closest_to(img, self.search_list)
        if found is None:
            return print('No matching pairs found')
        self.match_list.append({'color': found['img'], 'depth': img['depth']})
        print('Match_list'+str(len(self.match_list)))
        return found

    def crop_perm_area(self, found, img):
        """
        This method crops the colored images. Later on, the final net will need to identify the position of the Springmittel Minus.
        Therefore, it is necessary to eliminate part of the picture where the position of the Springmittel in the pallet cannot be identified anymore.
        At the same time depth image will be cropped according to the color image. The dimensions of the image were taken by eye for reasons of practicality.
        As soon as it is no longer possible for the human eye to recognise how a Springmittel is arranged, the pallet must be shifted.
        """

        perm_width_left = 310
        perm_width_right = 1060
        perm_height_top = 40
        perm_height_bottom = 690

        color = found
        depth = img

        color_cropped = color[perm_height_top:perm_height_bottom, perm_width_left:perm_width_right]
        depth_cropped = depth[int(perm_height_top / 2):int(perm_height_bottom / 2), int(perm_width_left / 2):int(perm_width_right / 2)]

        shape_color = color_cropped.shape
        shape_depth = depth_cropped.shape

        color_width = color_cropped.shape[1]
        color_height = color_cropped.shape[0]

        return color_cropped, depth_cropped, color_width, color_height

    def predict_pallet(self, color_cropped):

        with open(PALLET_LABELS_FILENAME, 'r') as f:
            labels = [l.strip() for l in f.readlines()]

        od_model = ONNXRuntimeObjectDetectionPallet(PALLET_MODEL_FILENAME, labels)
        predictions = od_model.predict_image(color_cropped)

        return predictions

    def crop_pallet(self, predictions, color_width, color_height, color_cropped, depth_cropped):

        pallet_depth_cropped = None
        pallet_color_cropped = None
        pallet_color_height = None
        pallet_color_width = None

        for i in range(0, len(predictions)):
            if len(predictions) == 0:
                return print("There was no pallet detected")

            #The processing of the image is interrupted when the pallet cannot be properly detected, threshold is 0.8
            elif predictions[i]['probability'] < 0.6:
                return print("Pallet cannot be properly detected")

            elif predictions[i]['probability'] > 0.6:
                left = int(predictions[i]['boundingBox']['left'] * color_width)
                top = int(predictions[i]['boundingBox']['top'] * color_height)
                right = int(left + predictions[i]['boundingBox']['width'] * color_width)
                bottom = int(top + predictions[i]['boundingBox']['height'] * color_height)

                pallet_color_cropped = color_cropped[top:bottom, left:right]
                pallet_depth_cropped = depth_cropped[int(top / 2):int(bottom / 2), int(left / 2):int(right / 2)]

                pallet_color_width = pallet_color_cropped.shape[1]
                pallet_color_height = pallet_color_cropped.shape[0]

                pallet_depth_width = pallet_depth_cropped.shape[1]
                pallet_depth_height = pallet_depth_cropped.shape[0]

                print('Pallet cropped')

        return pallet_depth_cropped, pallet_color_cropped, pallet_color_height, pallet_color_width

    def predict_springmittel(self, pallet_color_cropped):

        with open(SPRINGMITTEL_LABELS_FILENAME, 'r') as f:
            labels = [l.strip() for l in f.readlines()]

        od_model = ONNXRuntimeObjectDetectionSpringmittel(SPRINGMITTEL_MODEL_FILENAME, labels)
        predictions_springmittel = od_model.predict_image(pallet_color_cropped)

        return predictions_springmittel

    def crop_springmittel(self, predictions_springmittel, pallet_color_width, pallet_color_height, pallet_depth_cropped, pallet_color_cropped):

        springmittel_color_cropped = None
        springmittel_depth_cropped = None

        for i in range(0, len(predictions_springmittel)):
            if len(predictions_springmittel) == 0:
                print("There was no springmittel detected")

            elif predictions_springmittel[i]['probability'] < 0.3:
                print("Springmittel cannot be properly detected")

            elif predictions_springmittel[i]['probability'] > 0.3:
                left = int(predictions_springmittel[i]['boundingBox']['left'] * pallet_color_width)
                top = int(predictions_springmittel[i]['boundingBox']['top'] * pallet_color_height)
                right = int(left + predictions_springmittel[i]['boundingBox']['width'] * pallet_color_width)
                bottom = int(top + predictions_springmittel[i]['boundingBox']['height'] * pallet_color_height)

                springmittel_color_cropped = pallet_color_cropped[top:bottom, left:right]
                springmittel_depth_cropped = pallet_depth_cropped[int(top / 2):int(bottom / 2), int(left / 2):int(right/2)]
                print('Springmittel cropped')

            return springmittel_color_cropped, springmittel_depth_cropped

    def handle_cropped_img(self, springmittel_color_cropped, springmittel_depth_cropped):

        #{'time': timestamp, 'img': img}
        self.cropped_list.append({'color': springmittel_color_cropped, 'depth': springmittel_depth_cropped})
        print('cropped_list'+str(len(self.cropped_list)))
        #self.cropped_list.append([[springmittel_color_cropped], [springmittel_depth_cropped]])

        return self.cropped_list

    def find_entry_temporal_closest_to(self, reference_entry, search_list, threshold=0.5):
        """
        This method searches the passed list for an entry which time attribute is closest to the
        time of the reference entry and the distance between reference time and the entry time
        must be below a certain threshold. If no entry was found which fulfills these condition
        None will be returned. All entries of the search list must be a dictionary with a
        time key and a timestamp as value.

        :param reference_entry:
            The entry of which the time is used to determine if an entry of the list is close enough
        :param search_list:
            The list which will be searched for the closest entry
        :return: The entry which is closest to the reference and below threshold otherwise None will be returned
        """
        reference_time = reference_entry['time']
        closest_distance = threshold
        closest_entry = None
        for entry in search_list:
            entry_time = entry['time']
            time_difference = abs(entry_time - reference_time)
            if time_difference < threshold and time_difference < closest_distance:
                closest_entry = entry
                closest_distance = time_difference
        return closest_entry







