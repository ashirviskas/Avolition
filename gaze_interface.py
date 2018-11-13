from peyetribe import EyeTribe
from panda3d.core import *
from one_euro_filter import OneEuroFilter
import time

class GazeInterface:

    __filterConfig  = {
        'freq': 120,        # Hz
        'mincutoff': 0.2,   # Decreasing reduces jitter but increases lag. Select for slow movement
        'beta': 0.5,        # Increase to reduce high speed lag
        'dcutoff': 1.0      # this one should be ok
    }
    __oneEuroFilterX = None
    __oneEuroFilterY = None

    @staticmethod
    def __getFilter(new = False):
        if new == True or GazeInterface.__oneEuroFilterX is None:
            GazeInterface.__oneEuroFilterX = OneEuroFilter(**GazeInterface.__filterConfig)
            GazeInterface.__oneEuroFilterY = OneEuroFilter(**GazeInterface.__filterConfig)
        return (GazeInterface.__oneEuroFilterX, GazeInterface.__oneEuroFilterY)

    @staticmethod
    def reduceNoise(point):
        (fx, fy) = GazeInterface.__getFilter()
        timestamp = time.time()
        x = fx(point[0], timestamp)
        y = fy(point[1], timestamp)

        return Point2(x, y)

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

