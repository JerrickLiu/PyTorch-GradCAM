import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import fnmatch



def mask(path = '/Users/SirJerrick/Downloads/images'):
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
def main():
    mask()
    #final_image()
    #generate_colored_background()
if __name__ == '__main__':
    main()
