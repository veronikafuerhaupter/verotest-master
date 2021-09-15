from imagehandler import Imagehandler
from verotest.ros.ros import Ros
from imagecutter import Imagecutter
from time import time
import time
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
    timestamp = time.time()
    if found is None:
        return
    matches = {'color':found['img'], 'depth':img['depth'], 'time':timestamp}
    q.put(matches)

# get images from queue
def observationbuilder(): #Logik checken ob queue leer ist, if empty dann timeout logik #timeout wenn letztes match mehr als x sekunden her
    counter = 0
    time.sleep(100)
    while not q.empty():
        item = q.get()
        counter = counter + 1
        print('counter'+str(counter))
        color_cropped, depth_cropped, color_width, color_height = imagehandler.crop_perm_area(item['color'], item['depth'])
        predictions = imagehandler.predict_pallet(color_cropped)
        print(predictions[0]['probability'])
        if predictions is None:
            print('No Pallet detected')
        elif not predictions:
            print('No pallet detected')
        elif predictions[0]['probability'] > 0.9:
            print('Pallet detected')
            pallet_depth_cropped, pallet_color_cropped, pallet_color_height, pallet_color_width = imagehandler.crop_pallet(predictions, color_width, color_height, color_cropped, depth_cropped)
            predictions_springmittel = imagehandler.predict_springmittel(pallet_color_cropped)
            im = Image.fromarray(pallet_color_cropped)
            im.save('Pallette'+str(counter)+'.jpg')
            if predictions_springmittel is None:
                print('No Springmittel detected')
            elif not predictions_springmittel:
                print('No Springmittel detected')
            elif predictions_springmittel[0]['probability'] > 0.7:
                print('Springmittel detected')
                counter_springmittel = 1
                springmittel_color_cropped, springmittel_depth_cropped = imagehandler.crop_springmittel(predictions_springmittel, pallet_color_width, pallet_color_height, pallet_depth_cropped, pallet_color_cropped)
                cropped_list = imagehandler.handle_cropped_img(springmittel_color_cropped, springmittel_depth_cropped)
                #im = Image.fromarray(springmittel_color_cropped)
                #im.save('Springmittel_cropped'+str(counter_springmittel)+'.jpeg')
                counter_springmittel = counter_springmittel + 1
            else:
                print('No Springmittel')
    print(len(cropped_list))
    q.task_done()

def listrunner():
    final_image_list = imagecutter.create_observations(imagehandler.cropped_list)
    return final_image_list

# subscribe to ros topic
ros.subscribe_color_imgs(imagehandler.handle_color_img)
ros.subscribe_depth_imgs(put)

# turn-on the worker thread
observationbuilder()
print('All work completed')

# block until all tasks are done
listrunner()







