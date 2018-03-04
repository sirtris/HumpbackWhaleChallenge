import os
import matplotlib.pyplot as plt
import pandas as pd
import random

from skimage.color import rgb2gray
from skimage import exposure
from skimage import filters
from skimage import io

train_data = pd.read_csv('data/train.csv')

filename = os.path.join('data/train', train_data.Image[random.randrange(0,9850)])

imagecolor = io.imread(filename)
image = rgb2gray(imagecolor)
val = filters.threshold_otsu(image)

hist, bins_center = exposure.histogram(image)
plt.figure(figsize=(9, 4))
plt.subplot(131)
plt.imshow(image, cmap='gray', interpolation='nearest')
plt.axis('off')
plt.subplot(132)
plt.imshow(image < val, cmap='gray', interpolation='nearest')
plt.axis('off')
plt.subplot(133)
plt.plot(bins_center, hist, lw=2)
plt.axvline(val, color='k', ls='--')

plt.tight_layout()
plt.show()

