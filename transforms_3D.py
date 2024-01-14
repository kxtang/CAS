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
                print(axis)
                m = np.flip(m, axis)
        return m


class RandomRotate90:
    """
    Rotate an array by 90 degrees around the horizontal plane. Image should be 3D (length, width, height).

    When creating make sure that the provided RandomStates are consistent between raw and labeled datasets,
    otherwise the models won't converge.

    IMPORTANT: assumes (length, width, height) axis order, and perform the rotation on the (length, width) dimensions.
    """

    def __init__(self, seed, **kwargs):
        self.seed = seed
        # always rotate around z-axis
        self.axis = (0, 1)

    def __call__(self, m):
        np.random.seed(self.seed) 
        assert m.ndim == 3, 'Supports only 3D images'

        k = np.random.randint(0, 4)
        print(k)
        m = np.rot90(m, k, self.axis)
        return m

