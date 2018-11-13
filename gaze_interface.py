from peyetribe import EyeTribe
from panda3d.core import *


class GazeInterface():

    @staticmethod
    def frameToPoint2(frame):
        width = 1920
        height = 1080

        x = 0 if frame.avg.x < 0 else width if frame.avg.x > width else frame.avg.x
        y = 0 if frame.avg.y < 0 else height if frame.avg.y > height else frame.avg.y

        normalized_x = 2 * (x / width) - 1
        normalized_y = -1 * (2 * (y / height) - 1)

        return Point2(normalized_x, normalized_y)

    @staticmethod
    def getLastFrame(tracker):
        lastIdx = tracker._frameq.qsize()
        for i in range(lastIdx - 1):
            ef = tracker.next()
        ef = tracker.next(True)
        return ef
