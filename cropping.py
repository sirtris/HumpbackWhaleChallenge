import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import scipy
from sklearn.cluster import KMeans, DBSCAN
from sklearn import metrics
import pandas as pd
import random
import matplotlib.image as mpimg
import os
from multiprocessing import Process
import multiprocessing as mp
import concurrent.futures


dir = os.path.dirname(__file__)
TRAIN_PATH = dir + "/data/train/"
OUTPUT_PATH = dir + "/output_crp/"


def crop_image(filename, show_panels = False):
    # First, check if the file is already cropped:
    WHALE_ID = filename.split("/")[-1]
#    if os.path.isfile(OUTPUT_PATH + 'crp_' + WHALE_ID):
#        return

    dir = os.path.dirname(__file__)
    TRAIN_PATH = dir + "/data/train/"
    OUTPUT_PATH = dir + "/output_crp/"

    filename = TRAIN_PATH+filename

    img_file = Image.open(filename)
    img = img_file.load()

    [width, height] = img_file.size
    X=np.zeros(shape=(width, height, 3))
    if type(img[0,0]) is int:
        copy = np.zeros(shape=(width, height, 3))
        for x in range(0, width):
            for y in range(0, height):
                copy[x, y, :] = img[x,y]
        X = copy
        # print(X)
    else:
        for x in range(0, width):
            for y in range(0, height):
                X[x,y] = img[x,y]

    X = np.array(X)

    # print(np.reshape(X, newshape=(width * height, 3)))

    # ----------- Trying different clustering techniques: ----------------- #
    cl = "kmeans"
    nr_clusters = 2
    Y = []
    # K-means
    if cl == "kmeans":
        kmeans = KMeans(n_clusters=nr_clusters, random_state=0).fit(np.reshape(X, newshape=(width * height, 3)))
        Y = kmeans.predict(np.reshape(X, newshape=(width * height, 3)))
        Y = np.transpose(np.reshape(Y, (width, height)))


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

    """
    # nearest neighbour via majority vote

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
    Y = copy

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

    def mark_group2(row, col, group, area):
        frontier = list()
        frontier.append([row, col])
        while not frontier == []:
            [r, c] = frontier.pop(0)
            if r < 0 or r >= len(Y) or c < 0 or c >= len(Y[0]):
                continue
            if resp[r, c] != 0:
                continue
            if Y[r, c] != group:
                continue
            resp[r, c] = area
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

    out = open("bigfile.txt", 'w')
    for line in resp:
        # print(line[left:right], right-left, len(line[left:right]), line)
        for val in line:
            out.write(str(int(val)) + "\t")
        out.write("\n")
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
    reversed_area_ranking = list(reversed(sorted(area_dict.values())))
    whale_id = int(float([key for key in area_dict if area_dict[key] == reversed_area_ranking[1]][0]))

    # if that id is at the top left and top right, it cannot be the whale, so we chose the third biggest area:
    if resp[0,0] == whale_id and resp[0,len(resp[0])-1] == whale_id:
        whale_id = int(float([key for key in area_dict if area_dict[key] == reversed_area_ranking[2]][0]))

    # if that area makes less than 10% of the image, we chose just the biggest one:
    if area_dict[str(float(whale_id))]/(width*height) < 0.15:
        whale_id = int(float([key for key in area_dict if area_dict[key] == reversed_area_ranking[0]][0]))
    # print(whale_id)

    # update the responsibilities one final time, setting every pixel to 0, except the one from whale_id
    for index_row, row in enumerate(resp):
        for index_col, col in enumerate(row):
            if col != int(float(whale_id)):
                resp[index_row, index_col] = 0.0

    # now find 43the boundaries for the cropped picture:
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
    # print(left, right, top, bottom)

    cropped_img = np.zeros(shape=(bottom-top, right-left))
    for index_row, row in enumerate(Y[top:bottom]):
        for index_col, col in enumerate(row[left:right]):
            cropped_img[index_row, index_col] = col

    c_img = np.zeros(shape=(bottom-top, right-left))

    grayImage = Image.open(filename).convert('LA')
    pixels = list(grayImage.getdata())
    width, height = grayImage.size
    pixels = [pixels[i * width:(i + 1) * width] for i in range(height)]

    for i_y, y in enumerate(range(top, bottom)):
        for i_x, x in enumerate(range(left, right)):
            # print(i_x, i_y, len(X), len(X[0]), X.shape, c_img.shape)
            c_img[i_y, i_x] = pixels[y][x][0]

    if show_panels == True:
        plt.figure()
        plt.subplot(2, 2, 1)
        plt.imshow(mpimg.imread(filename))
        plt.title("actual image")
        plt.subplot(2, 2, 2)
        plt.title("kmeans prediction")
        plt.imshow(Y)
        plt.suptitle("Whale id: " + WHALE_ID)
        plt.subplot(2, 2, 3)
        plt.imshow(resp)
        plt.title("second biggest area vs rest")
        plt.subplot(2, 2, 4)
        plt.imshow(c_img, cmap='gray')
        plt.title("cropped version of image")
        plt.show()

    # The only thing left is to save the cropped image ie. c_img
    result = Image.fromarray(c_img.astype(np.uint8))
    result.save(OUTPUT_PATH + 'crp_' + WHALE_ID)
#    with open(OUTPUT_PATH+'crp_'+WHALE_ID,'wb') as file:
##        file.write(c_img)
##        np.save(file,c_img)
#        result.save(file)
#        file.close()

    return True

def main():

    # load all files and do automatic cropping:
    all_imgs = os.listdir(TRAIN_PATH)[0:20]

#    for progress, img in enumerate(all_imgs[0:20]):
    #    print(progress)
    #    p = Process(target=crop_image,args=(TRAIN_PATH+img,True))
    #    p.start()
    ##    p.join()
#        crop_image(img)

    with concurrent.futures.ProcessPoolExecutor() as executor:

        executor.map(crop_image,all_imgs)
#    pool = mp.Pool()
#    for file in all_imgs:

#    pool.map(crop_image,all_imgs)
##        print(res.get(timeout=1))


if __name__ == "__main__":
    # execute only if run as a script
    main()
