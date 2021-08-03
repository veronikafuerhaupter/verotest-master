from ros import Ros

Ros()
ros.subscribe_depth_img(test_fn)



# Lukas fragen, was es mit den Messages anstatt nodes auf sich hat!!! :D:D:D
def test_fn(img):
    depth_img = []
    ros = Ros()
    ros.subscribe_depth_img()
    print(img)
    # image_path = r'C:\Users\VeronikaF\Documents\Robin4lemi'
    # cv2.imwrite('2107_3_kiste_180_2.jpg', img)
    # ap = argparse.ArgumentParser()
    # ap.add_argument("-i", r'C:\Users\VeronikaF\Documents\Robin4lemi', required=True, help="path to the input image")
    # args = vars(ap.parse_args())

    # img.save('C:/Users/VeronikaF/Documents/Robin4lemi', 'JPEG')
    dataframe = pd.DataFrame(img)
    dataframe.to_csv(r'2107_3_180_2.csv')
    y = pd.read_csv(r'2107_3_180_2.csv')
    print(y)
    print(dataframe)
    print(img.shape)


# Methode, rosbag play, erstes image mit timestamp abspeichern in ...


ros = Ros()
ros.subscribe_depth_img(test_fn)

# ros.subscribe_depth_img(test_fn) #hier eigene Funktion übergeben, oder subsrive _depth_image
ros.spin()