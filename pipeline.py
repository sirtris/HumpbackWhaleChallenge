import pandas as pd
import os
import matplotlib.pyplot as plt
from glob import glob
import numpy as np
import keras_cnn_classifier as cnn
from os.path import split
from sklearn.preprocessing import LabelEncoder

#from keras.preprocessing.image import (
#    random_rotation, random_shift, random_shear, random_zoom,
#    random_channel_shift, transform_matrix_offset_center, img_to_array)
from PIL import Image

train_df = pd.read_csv('./data/train.csv')

# Get the list of train/test files
train = glob('data/train/*jpg')
test = glob('data/test/*jpg')

# train/test directories
train_dir = 'data/train/'
test_dir = 'data/test/'

## For testing purposes
# test dir
t = glob('test/*jpg')
t1 = 'test/'

# For resizing
idealWidth = 1050   
idealHeight = 600

# Augment a single image
def augment_image(file_name):
    # Open Image
    img = Image.open(file_name)

    # Augmentations
    img = img.convert('LA')
    img = img.resize((idealWidth, idealHeight))
    # image = shear(image)
    # image = shift(image)
    #  other transformations

    return np.array(img)[:, :, 0]

def fakeClassify1():
    solution = []
    for i in range(0,len(train_df)):
        imagename = train_df.Image[i]
        entry = [imagename, train_df.Id[i]]
        solution.append(entry)
    return solution

def combine(solutions):
    s = solutions.pop(0)
    result = np.array(s)
    for s in solutions:
        m = np.array([col[1] for col in s]) # array of second column of s
        result = np.append(result,np.reshape(m,(m.size,1)), axis=1) # add m to result
    return result
    
def csv(model):
# Write to csv and run on test set
    with open("data/submission.csv", "w") as f:
        f.write("Image,Id\n")
        for image in test:
            img = augment_image(image)
            x = img.astype("float32")
            # apply preprocessing to test images
            # x = image_gen.standardize(x.reshape(1, SIZE, SIZE))
            y = model.predict_proba(x.reshape(1, idealWidth, idealHeight, 1))
            predicted_args = np.argsort(y)[0][::-1][:5]
            predicted_tags = LabelEncoder().inverse_transform(predicted_args)
            image = split(image)[-1]
            predicted_tags = " ".join(predicted_tags)
            f.write("%s,%s\n" % (image, predicted_tags))
    
csv(cnn.run())