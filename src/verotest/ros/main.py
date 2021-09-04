from imagehandler import Imagehandler
from verotest.ros.ros import Ros
from imagecutter import Imagecutter
from time import time
from PIL import Image
from multiprocessing import Process
import threading, queue

ros = Ros()
imagehandler = Imagehandler()
imagecutter = Imagecutter()

q = queue.Queue()
exit_thread = threading.Event()

# put images in a queue
def put(img): #matching, matches in queue, timestamp setzen wenn match erfolgreich
    found = imagehandler.handle_depth_img(img)
    timestamp = time()
    if found is None:
        return
    matches = {'color':found['img'], 'depth':img['depth'], 'time':timestamp}
    q.put(matches)

# get images from queue
def observationbuilder(): #Logik checken ob queue leer ist, if empty dann timeout logik #timeout wenn letztes match mehr als x sekunden her
    while True:
        timestamp = time()
        if not q.empty():
            item = q.get()
            counter = 0
            color_cropped, depth_cropped, color_width, color_height = imagehandler.crop_perm_area(item['color'], item['depth'])
            predictions = imagehandler.predict_pallet(color_cropped)
            if len(predictions) == 0 or predictions[counter]['probability'] < 0.6 or predictions is None:
                return print("No pallet detected")
            print(predictions)
            pallet_depth_cropped, pallet_color_cropped, pallet_color_height, pallet_color_width = imagehandler.crop_pallet(predictions, color_width, color_height, color_cropped, depth_cropped)
            if pallet_color_cropped is None:
                return ('Pallet color is None')
            predictions_springmittel = imagehandler.predict_springmittel(pallet_color_cropped)
            if len(predictions_springmittel) == 0 or predictions_springmittel[counter]['probability'] < 0.6 or predictions_springmittel is None:
                return print("No springmittel detected")
            springmittel_color_cropped, springmittel_depth_cropped = imagehandler.crop_springmittel(predictions_springmittel, pallet_color_width, pallet_color_height, pallet_depth_cropped,pallet_color_cropped)
            if springmittel_color_cropped is None:
                return print('Springmittel color is None')
            cropped_list = imagehandler.handle_cropped_img(springmittel_color_cropped, springmittel_depth_cropped)
            final_image_list = imagecutter.create_observations(cropped_list)
            counter = counter + 1

# subscribe to ros topic
ros.subscribe_color_imgs(imagehandler.handle_color_img)
ros.subscribe_depth_imgs(put)
observationbuilder()

# block until all tasks are done
q.join()
print('All work completed')


"""def put(q):

def observationbuilder(q):

if __name__ == "__main__":
    t1 = threading.Thread(target=put, args=(q,))
    t2 = threading.Thread(target=observationbuilder, args=(q,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    print("Done!")

    #Turn-on observation builder
threading.Thread(target=observationbuilder, daemon=True).start()

#Put depth & color in Queue



def whatever(img):
    found = imagehandler.handle_depth_img(img)
    color_cropped, depth_cropped, color_width, color_height = imagehandler.crop_perm_area(found, img)
    predictions = imagehandler.predict_pallet(color_cropped)
    pallet_depth_cropped, pallet_color_cropped, pallet_color_height, pallet_color_width = imagehandler.crop_pallet(predictions, color_width, color_height, color_cropped, depth_cropped)
    predictions_springmittel = imagehandler.predict_springmittel(pallet_color_cropped)
    springmittel_color_cropped, springmittel_depth_cropped = imagehandler.crop_springmittel(predictions_springmittel, pallet_color_width, pallet_color_height, pallet_depth_cropped, pallet_color_cropped)
    cropped_list = imagehandler.handle_cropped_img(springmittel_color_cropped, springmittel_depth_cropped)
    final_image_list = imagecutter.create_observations(cropped_list)


ros.subscribe_color_imgs(imagehandler.handle_color_img)
ros.subscribe_depth_imgs(whatever)

ros.spin()
"""






