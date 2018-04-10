import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os
import random


train_data = pd.read_csv('data/train.csv')
filename = os.path.join('data/train', train_data.Image[random.randrange(0,9850)])

img = cv2.imread(filename,0)

def black_and_white(image):
    for i in range(0,len(image)):
        for j in range(0,len(image[0])):
            if(image[i][j] != 0.0):
                image[i][j] = 255
    return image

#Changed sigma to a higher value to reduce noise
def auto_canny(img, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(img)
 
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(img, lower, upper)
 
    # return the edged image
    return edged

# second arugment = 0 of cv2.imread() load image as gray 
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#blurred = cv2.GaussianBlur(img, (5, 5), 0) #Increased the filter size to (5,5)
blurred = cv2.bilateralFilter(img,12,75,75) #We should experiment a bit with the parameters here

# apply Canny edge detection using a wide threshold, tight
# threshold, and automatically determined threshold
wide = cv2.Canny(blurred, 10, 200)
tight = cv2.Canny(blurred, 225, 250)
auto = auto_canny(blurred)

#Try to cancel out noise with closing and close gaps with connected component analysis. This image is shown separately
kernel = np.ones((5, 5), np.uint8)
#closed = cv2.morphologyEx(tight, cv2.MORPH_CLOSE, kernel) #dilation and erosion in one function
closed = cv2.dilate(tight,kernel,iterations = 1)
tighter = cv2.connectedComponents(closed,np.array([1,2]))
eroded = cv2.erode(black_and_white(tighter[1]).astype(np.float32),kernel,iterations = 1)
cv2.imwrite('charts/edgestest.jpg', eroded) #"Tighter" image won't show up unless it's saved and imported again
tighter2 = cv2.imread('charts/edgestest.jpg')

# show the images
cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
cv2.namedWindow('Edges', cv2.WINDOW_NORMAL)
cv2.namedWindow('Closed Edges',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Original', 300, 200)
cv2.resizeWindow('Closed Edges', 300, 200)
cv2.resizeWindow("Edges", 900, 300)
cv2.imshow("Original", img)
cv2.imshow('Closed Edges', cv2.cvtColor(tighter2, cv2.COLOR_BGR2GRAY))
cv2.imshow("Edges", np.hstack([wide, tight, auto]))
cv2.waitKey(0)

#plt.subplot(121),plt.imshow(img,cmap = 'gray')
#plt.title('Original Image'), plt.xticks([]), plt.yticks([])
#plt.subplot(122),plt.imshow(edges,cmap = 'gray')
#plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

#plt.show()