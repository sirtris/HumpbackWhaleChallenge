import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import testClassifier as tc

#from keras.preprocessing.image import (
#    random_rotation, random_shift, random_shear, random_zoom,
#    random_channel_shift, transform_matrix_offset_center, img_to_array)
from PIL import Image

train_df = pd.read_csv('./data/train.csv')

# Get the list of train/test files
train = os.listdir('data/train')
test = os.listdir('data/test')


# train/test directories
train_dir = 'data/train/'
test_dir = 'data/test/'

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
        # image = shift(image)
        # other transformations

        img.save('altered/' + image, "JPEG")

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
    
# augment_images(train, traindir)
s1 = classify1() #testClassify()
s2 = s1
print(combine([s1,s2]))