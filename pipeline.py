import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import (
    random_rotation, random_shift, random_shear, random_zoom,
    random_channel_shift, transform_matrix_offset_center, img_to_array)
from PIL import Image

train_df = pd.read_csv('./data/train.csv')

# Get the list of train/test files
train = os.listdir('data/train')
test = os.listdir('data/test')


# train/test directories
train_dir = 'data/train/'
test_dir = 'data/test/'


## For testing purposes
# test dir
t = os.listdir('test')
t1 = 'test/'
# For resizing
idealWidth = 1050
idealHeight = 600


def augment_images(files, directory):
    for image in files:
        fpath = directory + image
        img = Image.open(fpath)

        # Augment images
        img = img.resize((idealWidth, idealHeight), Image.NEAREST)
        img = img.convert("L")
        #img = img.rotate(30, Image.NEAREST)

        # image = shear(image)
        # image = shfit(image)
        # other transformations

        img.save('altered/' + image, "JPEG")


augment_images(t, t1)