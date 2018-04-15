import os
#from PIL import Image
import pandas as pd
from sklearn import svm
from scipy import misc
from sklearn.model_selection import cross_val_score
from time import time
from sklearn.ensemble import RandomForestClassifier

dir = os.path.dirname(__file__)
TRAIN_PATH = os.path.join(dir, "../")# dir + "/data/train/"
TEST_AMOUNT = 500 #len(os.listdir(os.path.join(TRAIN_PATH,"cropped/")))
all_imgs = os.listdir(TRAIN_PATH)[0:TEST_AMOUNT]

"""
ValueError: Found input variables with inconsistent numbers of samples: [64636, 64681]
64636 = len(os.listdir(os.path.join(TRAIN_PATH,"cropped/"))) - 707
(64636 = 65343 - 707)
64681 = nr of id's
(64681 was 64636)
64681 = csv file
65379 = nr of cropped images
"""


"""
code for when the import data works:
from IO

"""

"""
SVM:
For   500 images it runs 2 seconds (28% acc)
For  1000 images it runs 5.5 seconds (31% acc)
For  2000 images it runs 15 seconds (28% acc)
For  4000 images it runs 48 seconds (28% acc)
For 10000 images it runs 324.7 seconds

RF: max depth = 5
For   500 images it runs 2 seconds
For  1000 images it runs 4 seconds
For  2000 images it runs 7.5 seconds
For  4000 images it runs 14.5 seconds (18% acc)
For 10000 images it runs 38.4 seconds (8% acc)

RF: max depth = 10
For 10000 images it runs 41.7 seconds (26% acc)
For 20000 images it runs 93 seconds (15% acc)
For 30000 images it runs 151 seconds (9% acc)

RF: max depth = 20
For 10000 images it runs 48 seconds (54% acc)
For 20000 images it runs 107.5 seconds (44.6% acc)
For 30000 images it runs 177 seconds (35% acc)
For 40000 images Memory error

RF: max depth = 30
For 10000 images it runs 51 seconds (61% acc)
For 20000 images Memory error
For 30000 images Memory error
For 40000 images Memory error

RF: max depth = 15
For 20000 images it runs 92 seconds (30% acc)
For 30000 images it runs 157 seconds (22% acc)
For 40000 images it runs 236 seconds (19% acc)
For 50000 images Memory error

"""


def get_original_features():
    original_features = {}
    data = [line.replace(";\n", "") for line in open(os.path.join(dir, "../IO/original_features.csv"), 'r').readlines()[1:]]
    for line in data:
        features = line.split(";")
        original_features[features[0].split(".")[0]] = {"height":     int(features[1]),
                                  "width":      int(features[2]),
                                  "dimension":  int(features[3]),
                                  "mean_rgb":   int(features[4]),
                                  "min_rgb":    int(features[5]),
                                  "max_rgb":    int(features[6])
                                  }
    return original_features


def extract_features(data, original_features = {}, path = TRAIN_PATH):
    original_features = get_original_features()
    features = []
    for img_path in data[0:TEST_AMOUNT]:
        feature_row = []
        img_full_path = path + img_path
        try:
            img_file = misc.imread(img_full_path)
            whale_id = img_path.split(".")[0].split("_")[1]
            if whale_id in original_features:
                feature_row += [original_features[whale_id]["height"], original_features[whale_id]["width"], original_features[whale_id]["dimension"], original_features[whale_id]["mean_rgb"], original_features[whale_id]["min_rgb"], original_features[whale_id]["max_rgb"]]
            else:
                feature_row += [0,0,0,0,0,0]
            shape = img_file.shape
            if len(shape) == 2:
                feature_row += [shape[0], shape[1], 1]
            else:
                feature_row += [shape[0], shape[1], shape[2]]
            feature_row += [img_file.mean(), img_file.min(), img_file.max()]
            features.append(feature_row)
        except:
            print("file", img_path, "does not exist.")
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


def extract_features_from_original(data):
    # This function does not need to be run again, except you want to generate the "original_features.csv" again
    out = open("../IO/original_features.csv", 'w')
    out.write("id;width;height;dimensions;mean_rgb;min_rgb;max_rgb;\n")
    for index, line in enumerate(data):
        print(index)
        features = extract_features([line])
        out.write(line + ";")
        for f in features[0]:
            out.write(str(int(f)) + ";")
        out.write("\n")


def run_SVM(data,original_features = {}):
    t0 = time()
    clf = svm.SVC(kernel='rbf')
    print("extract features...")
    if original_features != {}:
        features = extract_features(data, original_features)
    else:
        features = extract_features(data)
    train_df = pd.read_csv('../IO/aug.csv')
    labels = [label for pic, label in train_df.as_matrix()[0:TEST_AMOUNT]]
    if False:
        # This section did not use cross validation; training and testing set was the same
        print("start training...")
        clf.fit(features, labels)
        print("start predicting...")
        Y = clf.predict(features)
        print(eval_pred(Y, labels))

    print("Do cross validation:")
    scores = cross_val_score(clf, features, labels, cv=10)
    print(scores)
    print(sum(scores)/len(scores))
    t1 = time()
    print("time:", t1-t0)


def run_RF(data, original_features = {}):
    t0 = time()
    print("extract features...")
    if original_features != {}:
        features = extract_features(data, original_features)
    else:
        features = extract_features(data)
    train_df = pd.read_csv('../IO/aug.csv')
    labels = [label for pic, label in train_df.as_matrix()[0:TEST_AMOUNT]]
    clf = RandomForestClassifier(max_depth=15, random_state=0)
    print("start CV:")
    scores = cross_val_score(clf, features, labels, cv=10)
    print(scores)
    print(sum(scores) / len(scores))
    t1 = time()
    print("time:", t1 - t0)


def SVM(kernel = 'rbf'):
    return svm.SVC(kernel=kernel)


def RF(max_depth = 10, ):
    return RandomForestClassifier(max_depth=max_depth, random_state=0)

if __name__ == '__main__':
    # extract_features_from_original(all_imgs)
    original_features = get_original_features()
    run_RF(all_imgs, original_features)