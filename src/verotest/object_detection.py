from queue import Queue
from time import time

import cv2
import numpy as np

from object_detection.object_detector.object_detector import ObjectDetector

from object_detection.ros.ros import Ros

from object_detection.logger.logger import Logger

gray_image_queue = Queue()
depth_image_queue = Queue()
object_detection_queue = Queue()



def main():
    object_detector = ObjectDetector(gray_image_queue, depth_image_queue, object_detection_queue)
    object_detector.start()

    ros = Ros()
    ros.subscribe_gray_img(process_img)
    ros.subscribe_depth_img(depth_img_handler)
    logger = Logger('Main')
    logger.info('Object detection started')
    while not ros.is_shutdown():
        if not object_detection_queue.empty():
            detected_object = object_detection_queue.get()
            print('Object detected: ' + str(detected_object))
            ros.publish_detected_object_msg(detected_object)
    pass


def depth_img_handler(depth_img):
    depth_image_queue.put({'time': time(), 'img': depth_img})

#BILDER ANNEHMEN UND IN QUEUE SCHMEISSEN AN OBJECT DETECTOR WEITERGEBEN
def process_img(img):
    gray_image_queue.put({'time': time(), 'img': img})


if __name__ == '__main__':
    main()
