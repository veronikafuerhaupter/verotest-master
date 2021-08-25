import rospy


class Logger:
    prefix = None

    def __init__(self, prefix):
        self.prefix = prefix

    def _add_prefix(self, message):
        return '[' + self.prefix + '] ' + message

    def debug(self, message):
        rospy.logdebug(self._add_prefix(message))

    def info(self, message):
        rospy.loginfo(self._add_prefix(message))

    def warn(self, message):
        rospy.logwarn(self._add_prefix(message))

    def error(self, message):
        rospy.logerr(self._add_prefix(message))


