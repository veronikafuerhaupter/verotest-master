# import the necessary packages
import cv2
import imutils

class ShapeDetector:

    def __init__(self):
        pass

    def detect(self, c):
        circles = 0
        # initialize the shape name and approximate the contour
        shape = "unidentified"
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        # if the shape is a triange, it will have 3 vertices
        if len(approx) == 3:
            shape = "triangle"
        # if the shaoe has 4 vertices, it is either a square or a rectangle
        elif len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
            # a square will have an aspect ratio that is approximately
            # equal to one, otherwise, the shape is a rectangle
            if 0.95 >= ar <= 1.5:
                shape = "square"
            else:
                shape = "rectangle"
        # if the shape is a pentagon, it will have 5 vertices
        elif len(approx) == 5:
            shape = "pentagon"

        # otherwise, we assume the shape is a circle
        else:
            shape = "circle"

        # return the name of the shape
        return shape

