# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 10:18:44 2018

@author: From kaggle notebook by PolarBear and notebook from Lex Toumbourou

"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras.preprocessing.image as kim
from keras.backend import tf as ktf
#import os
from PIL import Image
from matplotlib import pyplot as plt
import random
#import copy
#%% Set parameters 
#figWidth = figHeight = 9
#numAugmentations = 4

#%% exposed function

def augment_all_images(numAugmentations=10):
    # used to load images
    trainFrame = pd.read_csv("data/train.csv")
    #trainFrame = trainFrame.head(100)
    # used to keep track of new images
    augmentedFrame = pd.DataFrame(columns=['Image', 'Id'])
    # count present images
    idCountFrame = trainFrame.groupby("Id",as_index = False)["Image"].count()
    idCountFrame = idCountFrame.rename(columns = {"Image":"numImages"})
    for ix, row in idCountFrame.iterrows():
        if row['numImages'] < numAugmentations:
            if ix % 10 == 0:
                print(ix)
            # augment image when needed
            partFrame = augment_by_Id(row['Id'], trainFrame, augmentedFrame, numAugmentations)
            augmentedFrame.append(partFrame)
    # return the new ids
    augmentedFrame.to_csv('./data/train/augmented/aug.csv', sep =';', index = False)
    return augmentedFrame

#augment_allimages()

#%% Load images
# Has to be addaped to the pipeline (not just for one certain image)

def augment_by_Id(_id_, trainFrame, augFrame, numAugmentations):
    # filteredFrame -> all images that correspond to the ID
    # print(_id_)
    filteredFrame = trainFrame.loc[trainFrame['Id'] == _id_]
    func_list = [rotate_image, shear_image, zoom_image]#, shift_image]
    for i in range(filteredFrame.shape[0], numAugmentations):
        #filteredFrame[0]['Id'] # not sure about this line
        imgFileName = filteredFrame.iloc[i % filteredFrame.shape[0]]['Image']
        chosenImage = Image.open('./data/train/' + imgFileName) #filteredFrame['Image'][i % filteredFrame.shape[0]])
        # print(imgFileName)
        augmentedImage = augment_image(chosenImage, func_list)
        plt.imsave('./data/train/augmented/' + imgFileName[:-4] + "_" + str(i) + ".jpg", augmentedImage)
        augFrame.loc[augFrame.shape[0]] = [imgFileName[:-4] + "_" + str(i) + ".jpg", _id_]
    return augFrame
    #augmentedFrame.loc[0] = ['sad' for n in range(2)]
    #convert image to array
    #imageArray = np.array(chosenImage)


def augment_image(chosenImage, func_list):
    imageArray = np.array(chosenImage)
    if len(imageArray.shape) == 2:
        arrays = [imageArray for _ in range(3)]
        imageArray = np.stack(arrays, axis=2)
    return random.choice(func_list)(imageArray)

def rotate_image(img_arr, rotationSize = 20): 
    return kim.random_rotation(img_arr,rotationSize, 
                row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest')
    
def shear_image(img_arr, givenIntensity = 5):
    return kim.random_shear(img_arr, intensity= givenIntensity, 
                 row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest')
    
def zoom_image(img_arr, zoomRangeWidth = 1.5, zoomRangeHeight = .7):
    return kim.random_zoom(img_arr, zoom_range=(zoomRangeWidth,zoomRangeHeight),
                row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest')
    
def shift_image(img_arr, widthRange = 0.1, heightRange = 0.3):
    return kim.random_shift(img_arr, wrg= widthRange, hrg= heightRange, 
                     row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest')


'''
#%%
fullImageFilename = './data/train/ff38054f.jpg'
chosenImage = Image.open(fullImageFilename)

#convert image to array
imageArray = np.array(chosenImage)
#%% make rotation
rotationSize = 30
rotatedImages = [kim.random_rotation(imageArray,rotationSize, 
                row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest')
                for _ in range(numAugmentations)]

plt.imsave('out.jpg', rotatedImages[0])

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

'''
if __name__ == '__main__':
    
    augment_all_images(15)
    