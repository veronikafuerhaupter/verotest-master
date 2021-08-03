# import the necessary packages
import argparse
import cv2
import imutils
import numpy as np

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help='path to input image')
args = vars(ap.parse_args())

# load the image
image = cv2.imread(args["image"])
output = image.copy()

# convert it to grayscale, blur it slightly and threshold it
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
invert = cv2.bitwise_not(thresh)
cv2.imwrite('crop2_thresh.png', thresh)

# find contours in the inverted image
#cnts = cv2.findContours(invert.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#cnts = imutils.grab_contours(cnts)
#sd = ShapeDetector()

# detect circles in the image
circles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, 1.0, 100, param2=1.5)
# ensure at least some circles were found
if circles is not None:
	# convert the (x, y) coordinates and radius of the circles to integers
	circles = np.round(circles[0, :]).astype("int")
	# loop over the (x, y) coordinates and radius of the circles
	for (x, y, r) in circles:
		# draw the circle in the output image, then draw a rectangle
		# corresponding to the center of the circle
		cv2.circle(output, (x, y), r, (0, 255, 0), 2)
		cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
	# show the output image
	cv2.imshow("output", np.hstack([image, output]))
	cv2.waitKey(0)

# loop over the contours
#for c in cnts:
	# compute the center of the contour
	#M = cv2.moments(c)
	#print(M)
	#cX = int(M["m10"] / M["m00"])
	#cY = int(M["m01"] / M["m00"])
	# draw the contour and center of the shape on the image
	#cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
	#cv2.circle(image, (cX, cY), 7, (255, 255, 255), -1)
	#cv2.putText(image, "center", (cX - 20, cY - 20),
		#cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

	# show the image
	# print(image)
	#cv2.imwrite('image_invert1.jpg', invert)
	#cv2.imshow("Image", image)
	#cv2.waitKey(0)
