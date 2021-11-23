# import the necessary packages
import numpy as np
import argparse
import cv2
import PIL

# load the image, clone it for output, and then convert it to grayscale
image = cv2.imread("img_pallet96.png")
output = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
invert = np.invert(thresh)

# detect circles in the image
print(image.shape())
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=3, minDist=700, param1=500, param2=40, minRadius=9, maxRadius=14)

# ensure at least some circles were found
if circles is not None:
	# convert the (x, y) coordinates and radius of the circles to integers
	circles = np.round(circles[0, :]).astype("int")

	# loop over the (x, y) coordinates and radius of the circles
	for (x, y, r) in circles:
		# draw the circle in the output image, then draw a rectangle
		# corresponding to the center of the circle
		cv2.circle(output, (x, y), r, (0, 255, 0), 2)
		cv2.rectangle(output, (x - 2, y - 2), (x + 2, y + 2), (0, 128, 255), -1)

	# show the output image
	cv2.imshow("output", np.hstack([image, output]))
	cv2.waitKey(0)

