import numpy as np
import cv2
import imutils
from imagehandler import Imagehandler
from PIL import Image
from shapedetector import ShapeDetector

# load the image and resize it to a smaller factor so that
# the shapes can be approximated better
image = cv2.imread('CroppedList1_sample.jpg')
resized = imutils.resize(image, width=300)
ratio = image.shape[0] / float(resized.shape[0])
print(image.shape)
shape_list = []

# invert image
#invert = np.invert(image)

# convert the resized image to grayscale, blur it slightly,
# and threshold it
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
invert = np.invert(thresh)
im = Image.fromarray(invert)
im.save('Springmittelinvert.jpg')

# find contours in the thresholded image and initialize the
# shape detector
cnts = cv2.findContours(invert.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
sd = ShapeDetector()

#output = image.copy
#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# detect circles in the image
#circles = cv2.HoughCircles(invert, cv2.HOUGH_GRADIENT, dp=1, minDist=10, param1=350)


# ensure at least some circles were found
# loop over the contours
for c in cnts:
	# compute the center of the contour, then detect the name of the
	# shape using only the contour
	M = cv2.moments(c)
	if M["m00"] > 0:
		cX = int((M["m10"] / M["m00"]) * ratio)
		cY = int((M["m01"] / M["m00"]) * ratio)
	shape = sd.detect(c)
	shape_list.append(shape)
	# multiply the contour (x, y)-coordinates by the resize ratio,
	# then draw the contours and the name of the shape on the image
	c = c.astype("float")
	c *= ratio
	c = c.astype("int")
	cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
	cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
	# show the output image
	cv2.imshow("Image", invert)
	#cv2.waitKey(0)









