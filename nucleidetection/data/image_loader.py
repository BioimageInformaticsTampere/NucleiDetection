#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 11:34:51 2019

@author: miravalkonen
"""

import numpy as np
from skimage import io
from skimage import img_as_float
from skimage.transform import resize


def load_image(imagefile: str, dataset_mpp: float, model_mpp: float) -> np.array:
    """Load image from disk

    :param imagefile: full path to image file
    :param dataset_mpp: Micrometers per pixel in actual data
    :param model_mpp: Micrometers per pixel in model

    :returns: NumPy array of image
    """

    # TODO: rename ds, ask Mira
    ds = model_mpp / dataset_mpp

    img = io.imread(imagefile)

    # TODO: scipy.misc.imresize had option 'bilinear', not sure of this
    # TODO: write test for this
    size_new = (int(img.shape[0] * (1 / ds)), int(img.shape[1] * (1 / ds)))
    img = img_as_float(resize(img[:, :, 0:3], size_new)).astype("float32")

    return img


def load_conf(imagefile: str) -> np.array:
    """Load image from disk with no processing

    :param imagefile: path to image

    :returns: NumPy array of image
    """

    img = img_as_float(io.imread(imagefile)).astype("float32")

    return img
