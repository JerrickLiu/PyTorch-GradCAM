import cv2
import os


def image_enhancer(path = '/home/jerrick/Documents/Projects/For_Jerrick/data/4in_test_asphalt/test/2S1'):
    i = 0
    for root, dirs, filename in os.walk(path):
        for file in filename:
            new_path = os.path.join(root, file)
            img = cv2.imread(new_path, 1)
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl,a,b))
            final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            i += 1
            cv2.imwrite('/home/jerrick/Documents/Projects/For_Jerrick/data/enhanced_4in_test_asphalt/test/2S1/enhance_4in_asphalt{}.png'.format(i), final)

image_enhancer()
