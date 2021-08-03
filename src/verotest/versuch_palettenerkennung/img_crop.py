import os
import sys
import cv2
import onnxruntime
import onnx
import numpy as np
from PIL import Image, ImageDraw
from object_detection_v2 import ObjectDetection
import tempfile

class ImgCrop:

    def crop_img(self, image):

        left =