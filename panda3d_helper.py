from panda3d.core import *


class PandaHelper:

    @staticmethod
    def targetInsideFrame(target, frame):
        bounds = frame.getBounds()
        targetPos = target.getPos(frame)
        if bounds[0] < 0:
            x_aligned = targetPos[0] > bounds[0] and targetPos[0] < 0
        else:
            x_aligned = targetPos[0] < bounds[0] and targetPos[0] > 0

        if bounds[3] < 0:
            y_aligned = targetPos[2] > bounds[3] and targetPos[2] < 0
        else:
            y_aligned = targetPos[2] < bounds[3] and targetPos[2] > 0

        if x_aligned and y_aligned:
            return True
        return False
