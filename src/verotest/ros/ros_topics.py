from enum import Enum

#Topics so in meinem ROS Workspace erstellen und darauf subscriben
class RosTopics(Enum):
    LEFT_COLOR_IMAGE = '/stereo/left/image_rect_color'
    STEREO_DEPTH_IMAGE = '/stereo/depth'

