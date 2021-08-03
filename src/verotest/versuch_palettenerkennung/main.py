import os
import sys
import cv2
import onnxruntime
import onnx
import numpy as np
from PIL import Image, ImageDraw
from object_detection_v2 import ObjectDetection
import tempfile

from onnxruntime_predict_v2 import ONNXRuntimeObjectDetection, MODEL_FILENAME, LABELS_FILENAME


def main(image_filename):
    # Load labels
    with open(LABELS_FILENAME, 'r') as f:
        labels = [l.strip() for l in f.readlines()]

    od_model = ONNXRuntimeObjectDetection(MODEL_FILENAME, labels)

    image = Image.open(image_filename)
    predictions = od_model.predict_image(image)
    img_convert = od_model.convert(image)
    predictions = od_model.predict_image(image)

    ratio2, ratio3, old_width, old_height = od_model.prepare_crop(image)

    for i in range(0, len(predictions)):
        if predictions[i]['probability'] > 0.3:
            print(predictions)
            left = predictions[i]['boundingBox']['left'] * old_width
            top = predictions[i]['boundingBox']['top'] * old_height
            right = left + predictions[i]['boundingBox']['width'] * old_width
            bottom = top + predictions[i]['boundingBox']['height'] * old_height
            print(left)
            print(top)
            print(right)
            print(bottom)
            im1 = img_convert.crop((left, top, right, bottom))
            im = np.asarray(im1)
            cv2.imwrite('crop3.png', im)
            return predictions

            # x = left * image.width / ObjectDetection.

            # x = in_x.preprocess()
    # print(predictions[0]['probability'])


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print('USAGE: {} image_filename'.format(sys.argv[0]))
    else:
        main(sys.argv[1])
