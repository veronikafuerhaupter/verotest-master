from logging import getLogger

import pandas as pd
import numpy as np
import rospy
from cv_bridge import CvBridge
import cv2
from PIL import Image
import argparse


from verotest.ros.ros_topics import RosTopics
from sensor_msgs.msg import Image


from verotest.logger.logger import Logger

bridge = CvBridge()


class Ros:
    publish_dict = None
    logger = None

    def __init__(self, node_name='object_detection'):
        self.logger = Logger(Ros.__name__)
        self.publish_dict = dict()
        rospy.init_node(node_name, anonymous=True)

    #Auf Topic subscriben
    def subscribe_gray_img(self, subscriber):
        self._subscribe(RosTopics.LEFT_COLOR_IMAGE.value, Image, subscriber, self.grayscale_image_msg_parser)

    #Auf Topic subscriben --> stereo/
    def subscribe_depth_img(self, subscriber):
        self._subscribe(RosTopics.DEPTH_IMAGE.value, Image, subscriber, self.depth_image_msg_parser)
        #print("whatever")

    def _subscribe(self, topic, message, subscriber, parser=None):
        rospy.Subscriber(topic, message, lambda msg: subscriber(parser(msg)) if parser is not None else subscriber(msg))
        self.logger.info('Start listening on topic ' + topic)

    def _publish(self, topic, message_type, message):
        if topic in self.publish_dict:
            self.publish_dict[topic].publish(message)
        else:
            self.publish_dict[topic] = rospy.Publisher(topic, message_type, queue_size=10)
            self.publish_dict[topic].publish(message)

    #Eventuell eigene Logik einbauen weil ich keine Ecken detecte sondern anderes
    #def publish_detected_object_msg(self, detected_object):
    #    object_corners = []
    #    for corner in detected_object['corners']:
    #        object_corners.append(ObjectCorner(x=corner[0], y=corner[1], z=corner[2]))
    #    detected_object_msg = DetectedObject(label=str(detected_object['id']), height=detected_object['height'], corners=object_corners)
    #    self._publish(RosTopics.DETECTED_OBJECT.value, DetectedObject, detected_object_msg)

    def to_detected_tag(self, tag_msg):
        if len(tag_msg.tags) == 0:
            return tag_msg
        x_halved = 1280 / 2
        y_halved = 960 / 2
        tag_x = tag_msg.tags[0].pose.pose.position.x
        tag_y = tag_msg.tags[0].pose.pose.position.y
        x = x_halved + tag_x * x_halved
        y = y_halved + tag_y * y_halved
        print('x: ' + str(x))
        print('y: ' + str(y))
        return tag_msg

    def grayscale_image_msg_parser(self, msg):
        return bridge.imgmsg_to_cv2(msg, 'passthrough')
        #return bridge.imgmsg_to_cv2(msg, 'rgb8')

        #image_path = r'C:\Users\VeronikaF\Documents\Robin4lemi'
        #cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        #return cv2.imwrite(image_path, cv_img)

    def depth_image_msg_parser(self, msg):
        return np.nan_to_num(bridge.imgmsg_to_cv2(msg, 'passthrough'))

    def is_shutdown(self):
        return rospy.is_shutdown()

    def spin(self):
        rospy.spin()


#Lukas fragen, was es mit den Messages anstatt nodes auf sich hat!!! :D:D:D
def depth_img_list_creation(img):

    depth_img = []
    depth_img.append(img)
    print(img)

def grey_img_processing()
    #image_path = r'C:\Users\VeronikaF\Documents\Robin4lemi'
    #cv2.imwrite('2107_3_kiste_180_2.jpg', img)
    #ap = argparse.ArgumentParser()
    #ap.add_argument("-i", r'C:\Users\VeronikaF\Documents\Robin4lemi', required=True, help="path to the input image")
    #args = vars(ap.parse_args())


    #img.save('C:/Users/VeronikaF/Documents/Robin4lemi', 'JPEG')
    dataframe = pd.DataFrame(img)
    dataframe.to_csv(r'2107_3_180_2.csv')
    y = pd.read_csv(r'2107_3_180_2.csv')
    print(y)
    print(dataframe)
    print(img.shape)


#Methode, rosbag play, erstes image mit timestamp abspeichern in ...

def gray_img_processing(img):




ros = Ros()
ros.subscribe_depth_img(depth_img_processing())
ros.subscribe_gray_img(gray_img_processing())

#ros.subscribe_depth_img(test_fn) #hier eigene Funktion übergeben, oder subsrive _depth_image
ros.spin()

