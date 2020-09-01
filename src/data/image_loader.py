#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 11:34:51 2019

@author: miravalkonen
"""

from skimage import io
from skimage import img_as_float
from scipy.misc import imresize


def load_image(imagefile, dataset_mpp, model_mpp):

    ds = model_mpp / dataset_mpp

    img = io.imread(imagefile)
    img = img_as_float(
        imresize(img[:, :, 0:3], 1 / ds, interp="bilinear", mode=None)
    ).astype("float32")

    return img


def load_conf(imagefile):

    img = img_as_float(io.imread(imagefile)).astype("float32")

    return img
