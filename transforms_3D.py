import importlib
import random

import numpy as np
import torch
from scipy.ndimage import rotate, map_coordinates, gaussian_filter, convolve
from skimage import measure
from skimage.filters import gaussian
from skimage.segmentation import find_boundaries

# WARN: use fixed random state for reproducibility; if you want to randomize on each run seed with `time.time()` e.g.
# GLOBAL_RANDOM_STATE = np.random.RandomState(47)


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, m):
        for t in self.transforms:
            m = t(m)
        return m


class RandomFlip:
    """
    Randomly flips the image across the given axes. Image should be 3D (length, width, height).

    When creating make sure that the provided seed are consistent between raw and labeled datasets,
    otherwise the models won't converge.
    """

    def __init__(self, seed, axis_prob=0.5):
        assert seed is not None, 'seed cannot be None'
        self.seed = seed
        self.axes = (0, 1, 2)
        self.axis_prob = axis_prob

    def __call__(self, m):
        np.random.seed(self.seed) 
        assert m.ndim == 3, 'Supports only 3D images'
        for axis in self.axes:
            if np.random.uniform() > self.axis_prob:
                m = np.flip(m, axis)
        return m


class RandomRotate90:
    """
    Rotate an array by 90 degrees around the horizontal plane. Image should be 3D (length, width, height).

    When creating make sure that the provided RandomStates are consistent between raw and labeled datasets,
    otherwise the models won't converge.
    """

    def __init__(self, seed, **kwargs):
        self.seed = seed
        self.axis = (0, 1)
        # always rotate around z-axis

    def __call__(self, m):
        np.random.seed(self.seed) 
        assert m.ndim == 3, 'Supports only 3D images'

        k = np.random.randint(0, 4)
        m = np.rot90(m, k, self.axis)
        return m
    

class RandomCrop:
    """
    Crop an array to be fit for the input of 3D-UNet. Image should be 3D (length, width, height).

    When creating make sure that the provided RandomCrop are consistent between raw and labeled datasets,
    otherwise the models won't converge.

    The default shape of RandomCrop is set to be (128, 128, 64)
    """

    def __init__(self, seed, length=128, width=128, height=64):
        self.seed = seed
        self.length = length
        self.width = width
        self.height = height
        # always rotate around z-axis
        self.axis = (0, 1)

    def __call__(self, m):
        np.random.seed(self.seed) 
        assert m.ndim == 3, 'Supports only 3D images'
        shape = m.shape
        x = np.random.randint(0, shape[0]-self.length)
        y = np.random.randint(0, shape[1]-self.width)
        z = np.random.randint(0, shape[2]-self.height)
        # m = np.rot90(m, k, self.axis)

        return m[x:x+self.length, y:y+self.width, z:z+self.height]

