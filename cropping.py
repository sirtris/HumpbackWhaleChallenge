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

dir = os.path.dirname(__file__)
#filename = dir + "/data/train/"
#
#img_file = Image.open(filename + "e9162eee.jpg")
#img = img_file.load()

train_data = pd.read_csv(dir + '/data/train.csv')

# filename = os.path.join(dir + '/data/train', train_data.Image[random.randrange(0, 9850)])
filename = os.path.join(dir + '/data/train', "ddba0df3.jpg")
print("file:", filename)

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
nr_clusters = 2
Y = []
# K-means
if cl == "kmeans":
    kmeans = KMeans(n_clusters=nr_clusters, random_state=0).fit(X)
    print(kmeans.cluster_centers_)
    Y = kmeans.predict(X)
    print(Y)
    Y = np.transpose(np.reshape(Y, (width, height)))
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(mpimg.imread(filename))
    plt.subplot(2, 2, 2)
    plt.imshow(Y)

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
        # plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),markeredgecolor='k', markersize=14)

        xy = X[class_member_mask & ~core_samples_mask]
        # plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    # plt.show()
    plt.imshow(np.transpose(np.reshape(labels, (width, height))))
    plt.show()

# do the actual cropping:

# find the boundaries of the whale tail:
#
# left = 999999
# right = 0
# top = 999999
# bottom = 0
# for row in Y:
#    row = list(row)
#    if row.index(1) < left:
#        left = row.index(1)
#    row = list(reversed(row))
#    if row.index(1) < right:
#        right = row.index(1)
#
# print(left, right)

# nearest neighbour via majority vote
window = 1
print(Y)
copy = np.zeros(shape=(len(Y), len(Y[0])))
for index_row, row in enumerate(Y):
    for index_col, col in enumerate(row):
        votes = [0] * nr_clusters
        for x in range(-window, window):
            for y in range(-window, window):
                if index_row + x < 0 or index_col + y < 0 or index_col + y >= len(row) or index_row + x >= len(Y):
                    continue
                else:
                    votes[Y[index_row + x][index_col + y]] += 1

        #        print(index_row)
        #        Y[index_row][index_col] = votes.index(max(votes))
        copy[index_row][index_col] = votes.index(max(votes))

plt.subplot(2, 2, 3)
plt.imshow(copy)

# do it a second time to smooth it out even more:
copy2 = np.zeros(shape=(len(copy), len(copy[0])))
for index_row, row in enumerate(copy):
    for index_col, col in enumerate(row):
        votes = [0] * nr_clusters
        for x in range(-window, window):
            for y in range(-window, window):
                if index_row + x < 0 or index_col + y < 0 or index_col + y >= len(row) or index_row + x >= len(copy):
                    continue
                else:
                    votes[int(copy[index_row + x][index_col + y])] += 1

        #        print(index_row)
        #        Y[index_row][index_col] = votes.index(max(votes))
        copy2[index_row][index_col] = votes.index(max(votes))

plt.subplot(2, 2, 4)
plt.imshow(copy2)
#plt.show()
# note: smoothing seems to work, but it does not really make the distinction between tail and not tail better. It
# remains questionable if we should really use that.

# Find the biggest area of one color:
Y = copy2
resp = np.zeros(shape=(len(Y), len(Y[0])))

def mark_group(row, col, val, group_nr):
    if row < 0 or row >= len(Y) or col < 0 or col >= len(Y[0]):
        return
    if resp[row, col] != 0:
        return
    if Y[row,col] == val:
        # mark the actual position
        resp[row,col] = group_nr
        # mark all the neigbours
        mark_group(row + 1, col, val, group_nr)
        mark_group(row, col + 1, val, group_nr)
        mark_group(row - 1, col, val, group_nr)
        mark_group(row, col - 1, val, group_nr)

area = 1
for index_row, row in enumerate(Y):
    for index_col, col in enumerate(row):
        if resp[index_row][index_col] == 0:
            # if that cell is not assigned a group yet, assign it to one:
            group_val = Y[index_row][index_col]
            mark_group(index_row, index_col, group_val, area)
            area += 1

out = open("bigfile.txt", 'w')
for line in resp:
    for val in line:
        out.write(str(int(val)) + " ")
    out.write("\n")
    #print(line)

# now we have a representation of all distinctive fields of the clustering results. Look in "bigfile.txt to see it;
# show now how big the 2 biggest areas are; 2 because the water is most likely to be the biggest one
area_dict = {}
for row in resp:
    for val in row:
        if str(val) in area_dict:
            area_dict[str(val)] += 1
        else:
            area_dict[str(val)] = 1

print(list(sorted(area_dict.values())))

# From here on we have to identify which area is the whale and crop that out.

