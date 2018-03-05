import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os
import random


train_data = pd.read_csv('data/train.csv')
filename = os.path.join('data/train', train_data.Image[random.randrange(0,9850)])

img = cv2.imread(filename,0)

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
blurred = cv2.GaussianBlur(img, (3, 3), 0)
 
# apply Canny edge detection using a wide threshold, tight
# threshold, and automatically determined threshold
wide = cv2.Canny(blurred, 10, 200)
tight = cv2.Canny(blurred, 225, 250)
auto = auto_canny(blurred)
 
# show the images
cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
cv2.namedWindow('Edges', cv2.WINDOW_NORMAL)  
cv2.resizeWindow('Original', 300, 200)
cv2.resizeWindow("Edges", 900, 300)
cv2.imshow("Original", img)
cv2.imshow("Edges", np.hstack([wide, tight, auto]))
cv2.waitKey(0)



#plt.subplot(121),plt.imshow(img,cmap = 'gray')
#plt.title('Original Image'), plt.xticks([]), plt.yticks([])
#plt.subplot(122),plt.imshow(edges,cmap = 'gray')
#plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

#plt.show()