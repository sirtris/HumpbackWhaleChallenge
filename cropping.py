import numpy as np
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.pyplot as plt
import colorsys
from PIL import Image
import scipy
from sklearn.cluster import KMeans, DBSCAN
from sklearn import metrics
import pandas as pd
import random
import matplotlib.image as mpimg



import os
#dir = os.path.dirname(__file__)
#filename = dir + "/data/train/"
#
#img_file = Image.open(filename + "e9162eee.jpg")
#img = img_file.load()

train_data = pd.read_csv('data/train.csv')

filename = os.path.join('data/train', train_data.Image[random.randrange(0,9850)])

img_file = Image.open(filename)
img = img_file.load()

[width, height] = img_file.size
X=[]

for x in range(0, width):
    for y in range(0, height):
        [r,g,b] = img[x,y]
        X.append([r,g,b])

X = np.array(X)

# ----------- Trying different clustering techniques: -----------------#
cl = "kmeans"
Y = []
# K-means
if cl == "kmeans":
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    print(kmeans.cluster_centers_)
    Y = kmeans.predict(X)
    print(Y)
    Y = np.transpose(np.reshape(Y, (width,height)))
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(mpimg.imread(filename))
    plt.subplot(1,2,2)
    plt.imshow(Y)
    plt.show()

if cl == "DBSCAN":
    db = DBSCAN(eps=0.3, min_samples=10).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    print('Estimated number of clusters: %d' % n_clusters_)

    # #############################################################################
    # Plot result

    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = X[class_member_mask & core_samples_mask]
        #plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),markeredgecolor='k', markersize=14)

        xy = X[class_member_mask & ~core_samples_mask]
        #plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    #plt.show()
    plt.imshow(np.transpose(np.reshape(labels, (width, height))))
    plt.show()


# do the actual cropping:

# find the boundaries of the whale tail:
#
#left = 999999
#right = 0
#top = 999999
#bottom = 0
#for row in Y:
#    row = list(row)
#    if row.index(1) < left:
#        left = row.index(1)
#    row = list(reversed(row))
#    if row.index(1) < right:
#        right = row.index(1)
#
#print(left, right)