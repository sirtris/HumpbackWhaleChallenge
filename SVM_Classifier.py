import os
from PIL import Image
import pandas as pd
from sklearn import svm
from scipy import misc
import numpy as np
from sklearn.model_selection import cross_val_score

dir = os.path.dirname(__file__)
TRAIN_PATH = dir + "/data/train/"
TEST_AMOUNT = 100
all_imgs = os.listdir(TRAIN_PATH)[0:TEST_AMOUNT]

clf = svm.SVC()


def extract_features(data):
    features = []
    for img_path in data:
        feature_row = []
        img_path = TRAIN_PATH + img_path
        img_file = misc.imread(img_path)
        shape = img_file.shape
        if len(shape) == 2:
            feature_row += [shape[0], shape[1], 1]
        else:
            feature_row += [shape[0], shape[1], shape[2]]
        feature_row += [img_file.mean(), img_file.min(), img_file.max()]
        features.append(feature_row)

    return features


def eval_pred(Y, labels):
    good = 0
    bad = 0
    for y, l in zip(Y,labels):
        if y == l:
           good += 1
        else:
            bad += 1
    return good, bad


def run(data):
    print("extract features...")
    features = extract_features(data)
    train_df = pd.read_csv('./data/train.csv')
    labels = [label for pic, label in train_df.as_matrix()[0:TEST_AMOUNT]]
    print("start training...")
    clf.fit(features, labels)
    print("start predicting...")
    Y = clf.predict(features)
    print(eval_pred(Y, labels))#, cross_val_score(clf, features, labels, cv = 2))



run(all_imgs)