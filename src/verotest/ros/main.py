from imagehandler import Imagehandler
from verotest.ros.ros import Ros


def test_fn(img):
    print(img)
    return img

ros = Ros()
imagehandler = Imagehandler()

ros.subscribe_color_imgs(imagehandler.handle_color_img)
ros.subscribe_depth_imgs(imagehandler.handle_depth_img)

ros.spin()

imagehandler.handle_color_img(test_fn)
imagehandler.handle_depth_img(test_fn)


#imagehandler.img_crop(test_fn)





