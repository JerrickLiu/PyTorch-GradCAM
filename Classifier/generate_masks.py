import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import fnmatch



def mask(path = '/Users/SirJerrick/Downloads/images'):
    i = 0
    for root, dirs, filename in os.walk(path):
        for file in filename:
            new_path = os.path.join(root, file)
            img = cv2.imread(new_path)
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            i += 1
            ret, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            cv2.imwrite('/Users/SirJerrick/Downloads/masks/apple_mask{}.png'.format(i), thresh)

mask()