import matplotlib.pyplot as plt
from feature_selection import *



def show_features(image):
    plt.subplot(231),plt.plot(histogram(image), cmap = 'gray')
plt.subplot(232),plt.imshow(images[1], cmap = 'gray')
plt.subplot(233),plt.imshow(images[1], cmap = 'gray')
plt.subplot(234),plt.imshow(images[1], cmap = 'gray')
plt.subplot(235),plt.imshow(images[1], cmap = 'gray')
plt.show()