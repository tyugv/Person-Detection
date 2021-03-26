from PIL import Image
from scipy.fftpack import dct
import numpy as np
from numpy.lib.stride_tricks import as_strided


def mean_pooling(A, kernel_size, stride=None, padding=0):
    if stride is None:
        stride = kernel_size

    # Padding
    A = np.pad(A, padding, mode='constant')

    # Window view of A
    output_shape = ((A.shape[0] - kernel_size) // stride + 1,
                    (A.shape[1] - kernel_size) // stride + 1)
    kernel_size = (kernel_size, kernel_size)
    A_w = as_strided(A, shape=output_shape + kernel_size,
                     strides=(stride * A.strides[0],
                              stride * A.strides[1]) + A.strides)
    A_w = A_w.reshape(-1, *kernel_size)

    # Return the result of pooling
    return A_w.mean(axis=(1, 2)).reshape(output_shape)


def gradient(img, stride):
    rez = []
    for i in range(0, len(img) - stride - 1, stride):
        arr1 = img[i:i + stride]
        arr2 = img[i + 1:i + stride + 1]
        rez.append(np.sum(np.abs(arr1 - arr2)))
    return rez


def dft(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    return magnitude_spectrum


def histogram(img, step=5):
    im = Image.fromarray(img)
    rez = im.histogram()
    return [np.max(rez[step * i: step * (i + 1)]) for i in range(256 // step)]


def make_dataset(images, fun, features_len):
    data = np.zeros((len(images), features_len))
    for i, image in enumerate(images):
        im = image * 255
        features = fun(im)
        if len(features.shape) > 1:
            data[i] = features.flatten()
        else:
            data[i] = features
