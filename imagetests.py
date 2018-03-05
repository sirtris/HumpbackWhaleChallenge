import numpy as np
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
from PIL import Image
import colorsys
import matplotlib.image as mpimg
import random
import collections as co
import scipy as sp

def import_RGB(location):
    im = Image.open(location)
    #im.show()
    return im

#Imported from https://stackoverflow.com/questions/7274221/changing-image-hue-with-python-pil
def shift_hue(arr, hout):
    r, g, b, a = np.rollaxis(arr, axis=-1)
    h, s, v = rgb_to_hsv(r, g, b)
    h = hout
    r, g, b = hsv_to_rgb(h, s, v)
    arr = np.dstack((r, g, b, a))
    return arr

def colorize(image, hue):
    """
    Colorize PIL image `original` with the given
    `hue` (hue within 0-360); returns another PIL image.
    """
    img = image.convert('RGBA')
    arr = np.array(np.asarray(img).astype('float'))
    new_img = Image.fromarray(shift_hue(arr, hue/360.).astype('uint8'), 'RGBA')
    return new_img

#Source: https://stackoverflow.com/questions/42045362/change-contrast-of-image-in-pil
def change_contrast(img, level):
    factor = (259 * (level + 255)) / (255 * (259 - level))
    def contrast(c):
        return 128 + factor * (c - 128)
    return img.point(contrast)

if (__name__ == '__main__'):
    rgb_to_hsv = np.vectorize(colorsys.rgb_to_hsv)
    hsv_to_rgb = np.vectorize(colorsys.hsv_to_rgb)

    test = import_RGB('data/train/00a29f63.jpg')
    test2 = test.convert('HSV' )#Convert image to HSV

    #CHANGE HUE
    test3 = colorize(test2,120) #image becomes greenish

    #CHANGE CONTRAST
    test3 = change_contrast(test3,40) #increases the contrast

    #test
    test3.show()
    #test2.save('test.png')
