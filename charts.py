"""
Get information chart about the whale image training data
Kaggle Club 2018
"""
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
import copy

def import_data():
    return 0

def main():
    #Gather basic statistics
    #Code partially taken from https://www.kaggle.com/mmrosenb/whales-an-exploration/notebook
    train_size = len(os.listdir('data/train'))
    print('Size of training set = ' + str(train_size))
    test_size = len(os.listdir('data/test'))
    print('Size of test set = ' + str(test_size))
    trainFrame = pd.read_csv("data/train.csv")
    unique_ids = len(trainFrame["Id"].unique())
    print('Nr of unique whale IDs = ' + str(unique_ids))

    #Get the number of images per category and plot it
    idCountFrame = trainFrame.groupby("Id",as_index = False)["Image"].count()
    idCountFrame = idCountFrame.rename(columns = {"Image":"numImages"})
    class_sizes = idCountFrame.values[:,1].tolist()
    print('Average class size = ' + str(sum(class_sizes)/unique_ids))
    plt.plot(range(0,unique_ids),sorted(class_sizes))
    plt.title('Class size distribution in the training set')
    plt.xlabel('classes')
    plt.ylabel('size')
    plt.savefig('class_sizes.png');plt.clf()
    #Plot nr of images per category without "New whale"
    plt.plot(range(0,unique_ids-1),sorted(class_sizes)[:-1])
    plt.title('Class size distribution in the training set without "New whale"')
    plt.xlabel('classes')
    plt.ylabel('size')
    plt.savefig('class_sizes2.png');plt.clf()

    #Plot logarithmic distribution of images per category
    idCountFrame["density"] = idCountFrame["numImages"] / np.sum(idCountFrame["numImages"])
    idCountFrame = idCountFrame.sort_values("density",ascending = False)
    idCountFrame["rank"] = range(idCountFrame.shape[0])
    idCountFrame["logRank"] = np.log(idCountFrame["rank"] + 1)
    plt.plot(idCountFrame["logRank"],idCountFrame["density"])
    plt.xlabel("$\log(Rank)$")
    plt.ylabel("Density")
    plt.title("$\log(Rank)$-Density Plot for Labels")
    plt.savefig('density.png');plt.clf()

    #Get the image sizes of the data and print the largest, smallest and average image size
    trainsizes = [];testsizes = []
    for name in os.listdir('data/train'):
        img = Image.open('data/train/' + name)
        trainsizes.append(img.size)
    for name in os.listdir('data/test'):
        img = Image.open('data/test/' + name)
        testsizes.append(img.size)
    breadth_train = np.array([x[0] for x in trainsizes])
    heigth_train = np.array([x[1] for x in trainsizes])
    breadth_test = np.array([x[0] for x in testsizes])
    heigth_test = np.array([x[1] for x in testsizes])
    print('Average training data image dimensions: (' + str(int(breadth_train.mean())) + ',' + str(int(heigth_train.mean()))+ ')')
    print('Average test data image dimensions: (' + str(int(breadth_test.mean())) + ',' + str(int(heigth_test.mean()))+ ')')
    print('Average training data standard deviation for image dimensions: (' + str(int(breadth_train.std())) + ',' + str(int(heigth_train.std()))+ ')')
    print('Average test data standard deviation for image dimensions: (' + str(int(breadth_test.std())) + ',' + str(int(heigth_test.std()))+ ')')

    #plot the train image sizes
    plt.plot(range(0,train_size),sorted([x[0] * x[1] for x in trainsizes]))
    plt.title('Image size distribution in the training set')
    plt.xlabel('images')
    plt.ylabel('size(pixels)')
    plt.savefig('imagesizetrain.png');plt.clf()
    #plot the test train image sizes
    plt.plot(range(0,test_size),sorted([x[0] * x[1] for x in testsizes]))
    plt.title('Image size distribution in the test set')
    plt.xlabel('images')
    plt.ylabel('size (pixels)')
    plt.savefig('imagesizetest.png');plt.clf()

if (__name__ == '__main__'):
    main()