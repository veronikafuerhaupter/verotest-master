import numpy as np


from os import listdir
from os.path import isfile, join


PATH = "/home/oem/Documents/Images/"

onlyfiles = [f for f in listdir(PATH) if "img_col" in f]

print(onlyfiles)

for counter, file in enumerate(onlyfiles):
    if counter == 0:
        image_all = np.load(file)
    else:
        image = np.load(file)
        image_resized = np.reshape(image, (90,90,3))
        image_all = np.concatenate((image_all, image_resized),axis = 2)
        print(image_all.shape)
        
print(image_all.shape)
        
