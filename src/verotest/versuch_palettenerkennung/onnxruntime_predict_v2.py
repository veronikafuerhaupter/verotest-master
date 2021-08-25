# The steps implemented in the object detection sample code: 
# 1. for an image of width and height being (w, h) pixels, resize image to (w', h'), where w/h = w'/h' and w' x h' = 262144
# 2. resize network input size to (w', h')
# 3. pass the image to network and do inference
# (4. if inference speed is too slow for you, try to make w' x h' smaller, which is defined with DEFAULT_INPUT_SIZE (in springmittel_detection.py or ObjectDetection.cs))
import os
import sys

import PIL.Image
import cv2
import onnxruntime
import onnx
import numpy as np
from PIL import Image, ImageDraw
from object_detection_v2 import ObjectDetection
import tempfile

MODEL_FILENAME = 'model.onnx'
LABELS_FILENAME = 'labels.txt'

class ONNXRuntimeObjectDetection(ObjectDetection):
    """Object Detection class for ONNX Runtime"""
    def __init__(self, model_filename, labels):
        super(ONNXRuntimeObjectDetection, self).__init__(labels)
        model = onnx.load(model_filename)
        with tempfile.TemporaryDirectory() as dirpath:
            temp = os.path.join(dirpath, os.path.basename(MODEL_FILENAME))
            model.graph.input[0].type.tensor_type.shape.dim[-1].dim_param = 'dim1'
            model.graph.input[0].type.tensor_type.shape.dim[-2].dim_param = 'dim2'
            onnx.save(model, temp)
            self.session = onnxruntime.InferenceSession(temp)
        self.input_name = self.session.get_inputs()[0].name
        self.is_fp16 = self.session.get_inputs()[0].type == 'tensor(float16)'
        
    def predict(self, preprocessed_image):
        inputs = np.array(preprocessed_image, dtype=np.float32)[np.newaxis,:,:,(2,1,0)] # RGB -> BGR
        inputs = np.ascontiguousarray(np.rollaxis(inputs, 3, 1))

        if self.is_fp16:
            inputs = inputs.astype(np.float16)

        outputs = self.session.run(None, {self.input_name: inputs})
        return np.squeeze(outputs).transpose((1,2,0)).astype(np.float32)


def main(image_filename):
    # Load labels
    with open(LABELS_FILENAME, 'r') as f:
        labels = [l.strip() for l in f.readlines()]

    od_model = ONNXRuntimeObjectDetection(MODEL_FILENAME, labels)

    #Image wird zur Liste --> def read_img merges col and depth img
    image = Image.open(image_filename)
    img_convert = od_model.convert(image)
    predictions = od_model.predict_image(image)
    old_width, old_height = od_model.prepare_crop(image)

    print(predictions)

    for i in range(0, len(predictions)):

        #Experimentations showed reasons for predictions greater 0.5 and smaller 0.8
        #are incorrectly placed boxes on the table,
        if predictions[i]['probability'] > 0.5:
            left = predictions[i]['boundingBox']['left'] * old_width
            top = predictions[i]['boundingBox']['top'] * old_height
            right = left + predictions[i]['boundingBox']['width'] * old_width
            bottom = top + predictions[i]['boundingBox']['height'] * old_height

            #Future life-implementation: Wait here and see whether pallet is moving

            print(left)
            print(top)
            print(right)
            print(bottom)
            im1 = img_convert.crop((left, top, right, bottom))
            im = np.asarray(im1)
            cv2.imwrite('crop3.png', im)
            return predictions

    return predictions
    return predictions, img_convert, old_width, old_height

def convert_predictions(self, predictions, img_convert, old_width, old_height):

    # Measures for desired camera area
    perm_width_left = 305
    perm_width_right = 1060
    perm_height_top = 40
    perm_height_bottom = 690

    for i in range(0, len(predictions)):

        # Experiments show that predictions with probability < 0.5 are most likely misqualified
        # Whereas the reasons for predictions with prob. > 0.5 mostly are pallets misplaced
        # and only a fractions of it shown in the camera area
        # As a result employees are asked to move the pallets into the desired position
        if predictions[i]['probability'] > 0.5:
            left = predictions[i]['boundingBox']['left'] * old_width
            if left < perm_width_left:
                print('Please move the pallet to the right')
                print('Bitte bewegen Sie die Palette nach rechts')

            right = left + predictions[i]['boundingBox']['width'] * old_width
            if right > perm_width_right:
                print('Please move the pallet to the left')
                print('Bitte bewegen Sie die Palette nach links')

            top = predictions[i]['boundingBox']['top'] * old_height
            if top < perm_height_top:
                print('Please move the pallet down')
                print('Bitte verschieben Sie die Pallette nach unten')

            bottom = top + predictions[i]['boundingBox']['height'] * old_height
            if bottom > perm_height_bottom:
                print('Please move the pallet up')
                print('Bitte verschieben Sie die Palette nach oben')

            # Future life-implementation: Wait here and see whether pallet is moving

            im1 = img_convert.crop((perm_width_left, perm_height_top, perm_width_right, perm_height_bottom))
            im = np.asarray(im1)
            cv2.imwrite('crop3.png', im)
            return im

        else: print('No object detected')


def img_crop(self, predictions, image_filename):

    perm_height_bottom = 40
    perm_height_top = 690
    perm_width_right = 305
    perm_width_left = 1060

    for i in range(0, len(predictions)):
        if predictions[i]['probability'] > 0.3:
            if predictions[i]['boundingBox']['left'] < perm_height_top:
                print('Schieb bitte die Palette nach links')
                print('Please move the pallet to the left')
            if predictions[i]['boundingBox']['left'] < perm_height_top:






    image = Image.open(image_filename)
    image = ObjectDetection.convert(image)

    left = predictions[0]['boundingBox']['left']
    top = predictions[0]['boundingBox']['top']
    right = left + predictions[0]['boundingBox']['width']
    bottom = top + predictions[0]['boundingBox']['height']

    im1 = image.crop((left, top, right, bottom))
    im1.show()


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print('USAGE: {} image_filename'.format(sys.argv[0]))
    else:
        main(sys.argv[1])
