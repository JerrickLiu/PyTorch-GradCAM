import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import fnmatch
import random


def mask(path = '/path/to/your/images/'):
    for root, dirs, filename in os.walk(path):
        for file in filename:
            new_path = os.path.join(root, file)
            img = cv2.imread(new_path)
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray_img, 20, 255, cv2.THRESH_BINARY_INV)
            f, e = os.path.splitext(new_path)
            cv2.imwrite(f + 'Masked.png', thresh)

def final_image(path = '/same/path/where/you/generated/the/masks'):
    for root, dirs, filename in os.walk(path):
        for file in sorted(filename):
            if fnmatch.fnmatch(file, '*.png') and not fnmatch.fnmatch(file, '*Masked.png'):
                new_path = os.path.join(root, file)
                img = cv2.imread(new_path)
            if fnmatch.fnmatch(file, '*Masked.png'):
                mask_path = os.path.join(root, file)
                mask = cv2.imread(mask_path)
                out = mask.copy()
                out[mask == 255] = img[mask == 255]
                f, e = os.path.splitext(mask_path)
                cv2.imwrite(f + 'MaskedFinal.png', out)

def generate_colored_background(path = '/same/path'):
    for root, dirs, filename in os.walk(path):
        for file in filename:
            if fnmatch.fnmatch(file, '*MaskedFinal.png'):
                new_path = os.path.join(root, file)
                color = list(np.random.choice(range(256), size=3))
                img = cv2.imread(new_path)
                img[np.where(((img == [0, 0, 0]).all(axis = 2)))] = color
                f, e = os.path.splitext(new_path)
                cv2.imwrite(f + 'ColoredBG.png', img)
                
def generate_image_as_background(path = '/same/path',
                             sand_path = '/path/where/the/image/you/want/as/background'):
    
#The background image must be the same size as your mask!! 

    images = []
    for root, dirs, filename in os.walk(sand_path):
        for file in filename:
            new_image_path = os.path.join(root, file)
            images.append(new_image_path)
    for root, dirs, filename in os.walk(path):
        for file in sorted(filename):
            bg_img = cv2.imread(random.choice(images))
            if fnmatch.fnmatch(file, '*.png') and not fnmatch.fnmatch(file, '*Masked.png') and not fnmatch.fnmatch(file,'*Final.png') and not fnmatch.fnmatch(file, '*BG.png'):
                new_path = os.path.join(root, file)
                img = cv2.imread(new_path)
            if fnmatch.fnmatch(file, '*Masked.png'):
                mask_path = os.path.join(root, file)
                masked_img = cv2.imread(mask_path)
                bg = cv2.bitwise_or(bg_img, masked_img, mask = None)
                out = bg.copy()
                out[bg == 255] = img[bg == 255]
                f, e = os.path.splitext(mask_path)
                cv2.imwrite(f + 'ImageBG.png', out)
def main():
    mask()
    #final_image()
    #generate_colored_background()
    #generate_image_as_background()
if __name__ == '__main__':
    main()
