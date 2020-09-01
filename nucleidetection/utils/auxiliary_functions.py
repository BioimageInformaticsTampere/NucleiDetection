#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 09:25:45 2019

@author: miravalkonen
"""

import os
import configparser
import logging

import numpy as np
import scipy
from scipy import ndimage as ndi
from scipy.ndimage import label, generate_binary_structure
from skimage.morphology import binary_dilation
import matplotlib.pyplot as plt
from skimage import img_as_ubyte

from nucleidetection.models import model
from nucleidetection.utils import constants

def run_nucdetect(config: configparser.ConfigParser) -> None:
    """Train or predict nuclei based on config file

    :param config: configparser.ConfigParser() object initialized with correct values
    :returns: None
    """
    logging.debug("Input path: " + config["imagepath"])
    logging.debug("Input imagetype: " + config["image_filetype"])
    logging.debug("Micrometers per pixel of the dataset: " + str(config.getfloat("dataset_mpp")))
    logging.debug("Detection results will be generated to path: " + config["outputpath"])
    logging.debug("MODE: " + config["MODE"])

    if not os.path.exists(config["outputpath"]):
        os.makedirs(config["outputpath"])

    if config["MODE"] == "detection":

        ## load trained model and predict nuc locations
        NucDetectNet = model.load_trained_model(
            config["model_path"], "cnnmodel"
        )
        model.predict_with_model(NucDetectNet, config)

    elif config["MODE"] == "adapt":

        ## tune the trained  model with new data
        model.domain_adapt(config)


def get_filelist(path, filetype):

    filelist = []
    for file in os.listdir(path):
        if file.endswith("".join([".", filetype])):
            filelist.append(file)

    return filelist


def vis_detections_from_coordinates(
    img, coordinates, name, savefig=False, outputpath=None, plotfig=False
):

    nucim = get_coordinate_overlay(img, coordinates, 4)
    if plotfig:
        plt.imshow(nucim)
    if savefig:
        filename = outputpath + name + ".tif"
        scipy.misc.imsave(filename, nucim)

    return nucim


def get_coordinates_from_confidence(conf, threshold, coordtype):

    conf = img_as_ubyte(conf)
    pred_binary = conf > threshold * 255
    s = generate_binary_structure(2, 5)
    labeled_array, nuccount = label(pred_binary, structure=s)
    coordinates = ndi.measurements.center_of_mass(
        pred_binary, labeled_array, range(1, nuccount)
    )
    coordinates = np.asarray(coordinates).astype(coordtype)

    return coordinates


def generate_binary_mask_from_coordinates(
    coordinates, image_size, markertype, markersize=1
):

    mask = np.zeros([image_size[0], image_size[1]], dtype="float32")
    for i in range(len(coordinates)):
        mask[coordinates[i][0], coordinates[i][1]] = 1

    if markertype == "coordinates":
        mask = mask

    elif markertype == "ball":
        selem = np.asarray(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )

        mask = binary_dilation(mask, selem=selem, out=None)
    return mask


def get_coordinate_overlay(orig_image, coordinates, mark_size=4):

    orig_image = img_as_ubyte(orig_image)
    R = orig_image[:, :, 0]
    G = orig_image[:, :, 1]
    B = orig_image[:, :, 2]

    step = int(mark_size / 2)

    for i in range(len(coordinates)):

        x, y = coordinates[i, :]
        R[x - step : x + step, y - step : y + step] = 0
        G[x - step : x + step, y - step : y + step] = 255
        B[x - step : x + step, y - step : y + step] = 0

    image = np.zeros(np.shape(orig_image), dtype="uint8")
    image[:, :, 0] = R
    image[:, :, 1] = G
    image[:, :, 2] = B

    return image
