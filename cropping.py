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
import scipy.misc

dir = os.path.dirname(__file__)
#filename = dir + "/data/train/"
#
#img_file = Image.open(filename + "e9162eee.jpg")
#img = img_file.load()

train_data = pd.read_csv(dir + '/data/train.csv')

filename = os.path.join(dir + '/data/train', train_data.Image[random.randrange(0, 9850)])
# filename = os.path.join(dir + '/data/train', "ddba0df3.jpg")
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
    print("clusters:", kmeans.cluster_centers_)
    Y = kmeans.predict(X)
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
"""
window = 3
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

plt.subplot(3, 2, 3)
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

plt.subplot(3, 2, 4)
plt.imshow(copy2)
#plt.show()
Y = copy2
"""
# note: smoothing seems to work, but it does not really make the distinction between tail and not tail better. It
# remains questionable if we should really use that. Especially because it takes forever to run.

# Find the biggest area of one color:

resp = np.zeros(shape=(len(Y), len(Y[0])))


def mark_group2(row, col, val, group_nr):
    frontier = list()
    frontier.append([row, col])
    while not frontier == []:
        [r, c] = frontier.pop(0)
        if r < 0 or r >= len(Y) or c < 0 or c >= len(Y[0]):
            continue
        if resp[r, c] != 0:
            continue
        if Y[r, c] != val:
            continue
        resp[r, c] = group_nr
        frontier.append([r + 1, c])
        frontier.append([r, c + 1])
        frontier.append([r - 1, c])
        frontier.append([r, c - 1])


def mark_group(row, col, val, group_nr):
    # recursive function to color the groups. Since the max recursion depth from python is easily exceeded, use
    # the other version
    if row < 0 or row >= len(Y) or col < 0 or col >= len(Y[0]):
        return
    if resp[row, col] != 0:
        return
    if Y[row, col] == val:
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
            mark_group2(index_row, index_col, group_val, area)
            area += 1

# ASSUMPTION: the pixel at the top left of the initial picture is not the whale tail.
water = Y[0,0]
for index_row, row in enumerate(Y):
    for index_col, col in enumerate(row):
        if Y[index_row, index_col] == water:
            resp[index_row, index_col] = 1


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

# print(list(reversed(sorted(area_dict.values()))))

# From here on we have to identify  which area the whale is inside and crop that out.
# For now, we will assume that the whale tail is the second biggest area:
whale_id = int(float([key for key in area_dict if area_dict[key] == list(reversed(sorted(area_dict.values())))[1]][0])) # Im sorry for that

print(whale_id)

# update the responsibilities one final time, setting every pixel to 0, except the one from whale_id
for index_row, row in enumerate(resp):
    for index_col, col in enumerate(row):
        if col != int(float(whale_id)):
            resp[index_row, index_col] = 0.0

plt.subplot(2, 2, 3)
plt.imshow(resp)
#plt.show()

# now find the boundaries for the cropped picture:
left, top = len(Y), len(Y)
right, bottom = len(Y), len(Y)
for index_row, row in enumerate(resp):
    if whale_id not in row:
        continue
    l = list(row).index(whale_id)
    if l < left:
        left = l
    r = list(reversed(row)).index(whale_id)
    if r < right:
        right = r

t_resp = np.transpose(resp)
for index_row, row in enumerate(t_resp):
    if whale_id not in row:
        continue
    t = list(row).index(whale_id)
    if t < top:
        top = t
    b = list(reversed(row)).index(whale_id)
    if b < bottom:
        bottom = b

right = len(Y[0]) - right
bottom = len(Y) - bottom
print(left, right, top, bottom)

cropped_img = np.zeros(shape=(bottom-top, right-left))
for index_row, row in enumerate(Y[top:bottom]):
    for index_col, col in enumerate(row[left:right]):
        cropped_img[index_row, index_col] = col

out = open("bigfile.txt", 'w')
for line in cropped_img:
    # print(line[left:right], right-left, len(line[left:right]), line)
    for val in line:
        out.write(str(int(val)) + " ")
    out.write("\n")

plt.subplot(2, 2, 4)
plt.imshow(cropped_img)
plt.show()

# The only thing left is to save the cropped image


# scipy.misc.toimage(cropped_img).save(dir + "/cropped_img/" + filename.split("\\")[1])

