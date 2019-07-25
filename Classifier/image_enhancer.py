import cv2
import os


def image_enhancer(path = '/path/of/images/you/want/enhanced'):
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
            cv2.imwrite('/folder/to/save/images/image{}.png'.format(i), final)

image_enhancer()
