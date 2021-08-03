from enum import Enum

#Topics so in meinem ROS Workspace erstellen und darauf subscriben
class RosTopics(Enum):
    #APRIL_TAG_DETECTED = '/rc_april_tag_detect/detected_tags'
    LEFT_COLOR_IMAGE = '/stereo/left/image_rect_color'
    #DETECTED_OBJECT = '/object_detection/detected_object'
    DEPTH_IMAGE = '/stereo/depth'