from queue import Queue
from threading import Thread
from time import time

import cv2
from verotest.logger.logger import Logger

from pupil_apriltags import Detector


class ObjectDetector(Thread):
    logger = None
    ref_des = None
    at_detector = None
    image_queue = None
    depth_queue = None
    detected_object_queue = None
    detected_tag_queue = None
    shutdown = False
    depth_history = None
    tags_history = None

    def __init__(self, image_queue, depth_queue, detected_object_queue):
        super(ObjectDetector, self).__init__()
        self.image_queue = image_queue
        self.depth_queue = depth_queue
        self.detected_object_queue = detected_object_queue
        self.detected_tag_queue = Queue()
        self.logger = Logger(ObjectDetector.__name__)
        self.depth_history = []
        self.tags_history = []

        self.at_detector = Detector(families='tag36h11',
                                    nthreads=1,
                                    quad_decimate=1.0,
                                    quad_sigma=0.0,
                                    refine_edges=1,
                                    decode_sharpening=0.25,
                                    debug=0)

    def run(self):
        Thread(target=self.detect_april_tags, args=(self.image_queue, self.detected_tag_queue)).start()
        while not self.shutdown:
            depth_tags = None
            if not self.detected_tag_queue.empty():
                self.tags_history.append(self.detected_tag_queue.get())
                depth_tags = self.merge_tag_and_depth(tags=self.tags_history[-1])
            if not self.depth_queue.empty():
                self.depth_history.append(self.depth_queue.get())
                depth_tags = self.merge_tag_and_depth(depth=self.depth_history[-1])
            if depth_tags is None:
                continue
            for tag in depth_tags['tags']:
                self.detected_object_queue.put({'time': depth_tags['time'], 'id': tag['id'], 'height': 0.3, 'corners': tag['corners']})


    def merge_tag_and_depth(self, tags=None, depth=None):
        if tags is None and depth is None:
            self.logger.info('No tag or depth found to merge')
            return
        if tags is None:
            tags = self.find_entry_temporal_closest_to(depth, self.tags_history)
            if tags is None:
                self.logger.info('No tag found which is temporal closer enough to the depth image')
                return
        if depth is None:
            depth = self.find_entry_temporal_closest_to(tags, self.depth_history)
            if depth is None:
                self.logger.info('No depth image found which is temporal closer enough to the detected tag')
                return

        merged_tags = []
        for tag in tags['tags']:
            depth_corners = []
            for corner in tag['corners']:

                depth_y = int(corner[1] / 2)
                depth_x = int(corner[0] / 2)
                depth_shape = depth['img'].shape
                if depth_y < 0 or depth_y >= depth_shape[0] or depth_x < 0 or depth_x >= depth_shape[1]:
                    depth_z = 0
                else:
                    depth_z = depth['img'][depth_y, depth_x]
                depth_corners.append((depth_x, depth_y, depth_z))
            merged_tags.append({'id': tag['id'], 'corners': depth_corners})
        return {'tags': merged_tags, 'time': depth['time']}

    # Hier könnte ich anfangen meine eigene Logik zu bauen


    def find_entry_temporal_closest_to(self, reference_entry, search_list):
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
        threshold = 0.5
        closest_distance = threshold
        closest_entry = None
        for entry in search_list:
            entry_time = entry['time']
            time_difference = abs(entry_time - reference_time)
            if time_difference < threshold and time_difference < closest_distance:
                closest_entry = entry
                closest_distance = time_difference
        return closest_entry


    def detect_april_tags(self, image_queue, detected_tags_queue):
        while not self.shutdown:
            if image_queue.empty():
                continue
            img_data = None
            while not image_queue.empty():
                #Aus queue rausholen
                img_data = image_queue.get()
                #Bild übergeben
            detections = self.at_detector.detect(img_data['img'], False, None, None)
            if len(detections) == 0:
                return []
            tags = []
            for detection in detections:
                tags.append({'id': detection.tag_id, 'corners': detection.corners})
            detected_tags_queue.put({'time': img_data['time'], 'tags': tags})


if __name__ == '__main__':
    detector = ObjectDetector()
