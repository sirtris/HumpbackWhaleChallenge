# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 10:18:44 2018

@author: From kaggle notebook by PolarBear and notebook from Lex Toumbourou

"""

import numpy as np # linear algebra
#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras.preprocessing.image as kim
from keras.backend import tf as ktf
#import os
from PIL import Image
from matplotlib import pyplot as plt
#import copy
#%% Set parameters 
figWidth = figHeight = 9
numAugmentations = 4

#%% Load images
# Has to be addaped to the pipeline (not just for one certain image)
fullImageFilename = f'./data/train/ff38054f.jpg'
chosenImage = Image.open(fullImageFilename)

#convert image to array
imageArray = np.array(chosenImage)


rotationSize = 30

#%% make rotation
rotatedImages = [kim.random_rotation(imageArray,rotationSize, 
                row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest')
                for _ in range(numAugmentations)]

# just needed for demonstration purposes
fig, givenSubplots = plt.subplots(2,2)
fig.set_size_inches(figWidth,figHeight)
for i in range(len(rotatedImages)):
    givenSubplots[int(i / 2),i % 2].imshow(rotatedImages[i])


#%% Random shift
numShifts = numAugmentations
widthRange = 0.1
heightRange = 0.3
shiftedImages = [kim.random_shift(imageArray, wrg= widthRange, hrg= heightRange, 
                 row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest')
                 for _ in range(numShifts)]

# Plot for demonstration purposes
fig, givenSubplots = plt.subplots(2,2)
fig.set_size_inches(figWidth,figHeight)
for i in range(len(shiftedImages)):
    givenSubplots[int(i / 2),i % 2].imshow(shiftedImages[i])
# -> parts of the whale tale might beshifted out of the image
    
#%% Random Shear
# Whatever a shear does. Not even the author of the notebook knows
# This should be it: http://www.enfocus.com/manuals/UserGuide/PP/10/enUS/assets/5844.png
# not sure if this is really working
numShears = numAugmentations
givenIntensity = 5      # was 0.4; but 5 provides more 'seeable' results; maybe even crank it up more
shearedImages = [kim.random_shear(imageArray, intensity= givenIntensity, 
                 row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest')
                 for _ in range(numShears)]

# Plot for demonstration purposes
fig, givenSubplots = plt.subplots(2,2)
fig.set_size_inches(figWidth,figHeight)
for i in range(len(shearedImages)):
    givenSubplots[int(i / 2),i % 2].imshow(shearedImages[i])
    
#%% Random Zoom
numZooms = numAugmentations
zoomRangeWidth = 1.5
zoomRangeHeight = .7
zoomedImages = [kim.random_zoom(imageArray, zoom_range=(zoomRangeWidth,zoomRangeHeight),
                row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest')
    for _ in range(numZooms)]
# -> part of the image might be out of frame

# Plot for demonstration purposes
fig, givenSubplots = plt.subplots(2,2)
fig.set_size_inches(figWidth,figHeight)
for i in range(len(zoomedImages)):
    givenSubplots[int(i / 2),i % 2].imshow(zoomedImages[i])
    
#%% pipeline for augmentation
def augmentation_pipeline(img_arr):
    img_arr = kim.random_rotation(img_arr,rotationSize, 
                row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest')
    img_arr = kim.random_shear(img_arr, intensity= givenIntensity, 
                 row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest')
    img_arr = kim.random_zoom(img_arr, zoom_range=(zoomRangeWidth,zoomRangeHeight),
                row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest')
    img_arr =kim.random_shift(img_arr, wrg= widthRange, hrg= heightRange, 
                     row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest')
    return img_arr

#%% test pipeline
imgs = [augmentation_pipeline(imageArray) for _ in range(4)]

fig, givenSubplots = plt.subplots(2,2)
fig.set_size_inches(figWidth,figHeight)
for i in range(len(imgs)):
    givenSubplots[int(i / 2),i % 2].imshow(imgs[i])
