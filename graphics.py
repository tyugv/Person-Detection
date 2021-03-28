import matplotlib.pyplot as plt
from feature_selection import *


def show_features(image):
    fig, axs = plt.subplots(2, 3, figsize=(9, 8))
    axs[0, 0].imshow(image), axs[0, 0].set_title('Оригинал')
    axs[0, 1].plot(histogram(image)), axs[0, 1].set_title('Гистограмма')
    axs[0, 2].imshow(dft(image), cmap='gray'), axs[0, 2].set_title('DFT')
    axs[1, 0].imshow(dct(image), cmap='gray'), axs[1, 0].set_title('DCT')
    axs[1, 1].imshow(mean_pooling(image, 4), cmap='gray'), axs[1, 1].set_title('Scale')
    axs[1, 2].plot(gradient(image, 2)), axs[1, 2].set_title('Градиент')
    fig.show()


def show_features_progress():
    titles = ['Мнение большинства', 'Градиент', 'Scale', 'DCT', 'DFT', 'Гистограмма']
    fig, axs = plt.subplots(2, 3, figsize=(9, 8))
    for i in range(2):
        for j in range(3):
            axs[i, j].set_xlim(1, 9)
            axs[i, j].set_ylim(0, 1)
            axs[i, j].set_title(titles.pop())
    return fig, axs