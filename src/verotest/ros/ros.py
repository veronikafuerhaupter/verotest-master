from os import environ
from time import time

import rospy
from cv_bridge import CvBridge
from verotest.logger.logger import Logger
from verotest.ros.ros_topics import RosTopics
from sensor_msgs.msg import Image

bridge = CvBridge()


class Ros:
    publish_dict = None
    logger = None

    def __init__(self, node_name='gesture_systems'):
        self.logger = Logger(Ros.__name__)
        self.publish_dict = dict()
        if 'DEBUG' in environ:
            rospy.init_node(node_name, anonymous=True, log_level=rospy.DEBUG)
        else:
            rospy.init_node(node_name, anonymous=True)

    def subscribe_color_imgs(self, subscriber):
        self._subscribe(RosTopics.LEFT_COLOR_IMAGE.value, Image, subscriber, self.color_image_msg_parser)

    def subscribe_depth_imgs(self, subscriber):
        self._subscribe(RosTopics.STEREO_DEPTH_IMAGE.value, Image, subscriber, self.stereo_depth_msg_parser)

    def _subscribe(self, topic, message, subscriber, parser=None):
        rospy.Subscriber(topic, message, lambda msg: subscriber(parser(msg)) if parser is not None else subscriber(msg), queue_size=3)
        self.logger.debug('Start listening on topic ' + topic)

    def _publish(self, topic, message_type, message):
        if topic in self.publish_dict:
            self.publish_dict[topic].publish(message)
        else:
            self.publish_dict[topic] = rospy.Publisher(topic, message_type, queue_size=10)
            self.publish_dict[topic].publish(message)

    def color_image_msg_parser(self, msg):
        timestamp = time()
        img = bridge.imgmsg_to_cv2(msg, 'rgb8')
        return {'time': timestamp, 'img': img}

    def stereo_depth_msg_parser(self, msg):
        timestamp = time()
        return {'time': timestamp, 'depth': bridge.imgmsg_to_cv2(msg, 'passthrough')}

    def is_shutdown(self):
        return rospy.is_shutdown()

    def spin(self):
        rospy.spin()


def test_fn(img):
    print(img)
    return img

initiator = Ros()
initiator.subscribe_color_imgs(test_fn)
