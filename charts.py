"""
Get information chart about the whale image training data
Kaggle Club 2018
"""
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
import matplotlib.image as mpimg
import random
from PIL import Image
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
    plt.savefig('charts/class_sizes.png')

    #Import the data
    dataset = import_data()


if (__name__ == '__main__'):
    main()