from imagehandler import Imagehandler
from verotest.ros.ros import Ros
from imagecutter import Imagecutter
from time import time
import time
from PIL import Image
from multiprocessing import Process
import threading
import queue
import numpy as np
import argparse
import cv2

ros = Ros()
imagehandler = Imagehandler()
imagecutter = Imagecutter()

q = queue.Queue()
exit_thread = threading.Event()

# put images in a queue
def put(img): #matching, matches in queue, timestamp setzen wenn match erfolgreich
    found = imagehandler.handle_depth_img(img)
    timestamp = time.time()
    if found is None:
        return
    matches = {'color':found['img'], 'depth':img['depth'], 'time':timestamp}
    q.put(matches)

# get images from queue
def observationbuilder(): #Logik checken ob queue leer ist, if empty dann timeout logik #timeout wenn letztes match mehr als x sekunden her
    counter = 0

    # sleep to fill the queue with matches from the list
    time.sleep(120)

    # take every item from the queue for processing
    while not q.empty():
        item = q.get()
        print('counter'+str(counter))

        # crop permitted area in order to get only images that can be identified with its alignment
        color_cropped, depth_cropped, color_width, color_height = imagehandler.crop_perm_area(item['color'], item['depth'])

        # predict the pallet from the permitted area via Pallet Model from Azure ML
        predictions = imagehandler.predict_pallet(color_cropped)
        print(predictions[0]['probability'])

        if predictions is None:
            print('No Pallet detected')
            continue
        # further processing if pallet detected in permitted area
        elif predictions[0]['probability'] > 0.9:
            print('Pallet detected')
            pallet_depth_cropped, pallet_color_cropped, pallet_color_height, pallet_color_width = imagehandler.crop_pallet(predictions, color_width, color_height, color_cropped, depth_cropped)

            #im = Image.fromarray(pallet_color_cropped)
            #im.save("Pallet"+str(counter)+".jpeg")

            # predict if Springmittel is in pallet via the inner radius
            circles = imagehandler.predict_circles(pallet_color_cropped)
            if circles is None:
                print('circles none')
                continue

            # Doublecheck with Springmittel prediction model from Azure ML
            predictions_springmittel = imagehandler.predict_springmittel(pallet_color_cropped)
            print(predictions_springmittel)

            if len(predictions_springmittel) == 0 or predictions_springmittel is None:
                print('No Springmittel')
                continue

            if circles is not None and predictions_springmittel[0]['probability'] > 0.8:
                # Crop predicted Springmittel in the pallet
                springmittel_color_cropped, springmittel_depth_cropped = imagehandler.crop_springmittel_circles(circles,pallet_color_cropped, pallet_depth_cropped)
                cropped_list = imagehandler.handle_cropped_img(springmittel_color_cropped, springmittel_depth_cropped)
                im = Image.fromarray(springmittel_color_cropped)
                im.save("Springmittel_cropped"+str(counter)+".jpeg")
                counter = counter + 1

            else:
                print('No Springmittel')
                continue
        else:
            print('Pallet not properly detectable')
            continue

    print(len(cropped_list))
    q.task_done()

def listrunner():
    outlierfree_list = imagecutter.remove_outlier(imagehandler.cropped_list)
    final_image_list = imagecutter.create_observations(outlierfree_list)
    return final_image_list

# subscribe to ros topic
ros.subscribe_color_imgs(imagehandler.handle_color_img)
ros.subscribe_depth_imgs(put)

# turn-on the worker thread
observationbuilder()
print('All work completed')

# block until all tasks are done
listrunner()







