import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

img = cv2.imread('/Users/SirJerrick/Downloads/images/apple3.jpeg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

plt.imshow(img)
#plt.show()

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors

pixel_colors = img.reshape((np.shape(img)[0]*np.shape(img)[1], 3))
norm = colors.Normalize(vmin=-1.,vmax=1.)
norm.autoscale(pixel_colors)
pixel_colors = norm(pixel_colors).tolist()

# h, s, v = cv2.split(hsv_img)
#
# fig = plt.figure()
# axis = fig.add_subplot(1, 1, 1, projection="3d")
#
from matplotlib.colors import hsv_to_rgb
#
# axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
# axis.set_xlabel("Hue")
# axis.set_ylabel("Saturation")
# axis.set_zlabel("Value")
# plt.show()

high_color = (400, 350, 350)
low_color = (0, 30, 30)

# lo_square = np.full((10, 10, 3), high_color, dtype=np.uint8) / 255.0
# do_square = np.full((10, 10, 3), low_color, dtype=np.uint8) / 255.0
# plt.subplot(1, 2, 1)
# plt.imshow(hsv_to_rgb(do_square))
# plt.subplot(1, 2, 2)
# plt.imshow(hsv_to_rgb(lo_square))
# plt.show()

mask = cv2.inRange(hsv_img, low_color, high_color)

result = cv2.bitwise_and(img, img, mask=mask)

blur = cv2.GaussianBlur(result, (7, 7,), 0)
plt.imshow(blur)
#plt.show()

def mask(path = '/Users/SirJerrick/Downloads/images', low_color =(0, 30, 30),
         high_color = (400, 350, 350)):
    i = 0
    for root, dirs, filename in os.walk(path):
        for file in filename:
            new_path = os.path.join(root, file)
            img = cv2.imread(new_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            mask = cv2.inRange(hsv_img, low_color, high_color)

            result = cv2.bitwise_and(img, img, mask = mask)

            blur = cv2.GaussianBlur(result, (7, 7), 0)
            plt.imshow(blur)
            plt.axis('off')
            plt.savefig('/Users/SirJerrick/Downloads/images/apple_mask{}'.format(i))
            plt.show()
            i += 1


mask()
