#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 09:25:45 2019

@author: miravalkonen
"""

import os
import numpy as np
import scipy
from scipy import ndimage as ndi
from scipy.ndimage import label, generate_binary_structure
from skimage.morphology import binary_dilation
import matplotlib.pyplot as plt
from skimage import img_as_ubyte

from NucleiDetection.detect import cnn_functions


class ProjectModel:
    pass


def run_nucdetect(arguments):

    cwd = os.getcwd()
    project_specs = ProjectModel()
    project_specs.projectpath = cwd + os.path.sep + "model" + os.path.sep
    project_specs.model_mpp = 0.5
    project_specs.batchoverlap = True
    project_specs.masktype = "ball"
    project_specs.mask_element_size = 2
    project_specs.block_size = 64
    project_specs.batch_size = 128
    project_specs.nb_epoch = 1
    project_specs.network_ds = 1

    for i in range(len(arguments)):
        substr = arguments[i].split(":")

        if substr[0] == "MODE":
            mode = substr[1]

        if substr[0] == "imagepath":
            project_specs.imagepath = substr[1]
        if substr[0] == "image_filetype":
            project_specs.image_filetype = substr[1]
        if substr[0] == "gtpath":
            project_specs.gtpath = substr[1]
        if substr[0] == "gt_filetype":
            project_specs.gt_filetype = substr[1]
        if substr[0] == "dataset_mpp":
            project_specs.dataset_mpp = float(substr[1])
        if substr[0] == "outputpath":
            project_specs.outputpath = substr[1]
        if substr[0] == "mask_identifier_suffix":
            project_specs.mask_identifier_suffix = substr[1]
        if substr[0] == "mask_identifier_prefix":
            project_specs.mask_identifier_prefix = substr[1]
        if substr[0] == "output_mode":
            multioutputbool = substr[1].find(",")

            if multioutputbool == -1:
                project_specs.output_mode = [substr[1]]
            else:
                outputs = substr[1].split(",")
                project_specs.output_mode = outputs

    print("Input path: " + project_specs.imagepath)
    print("Input imagetype: " + project_specs.image_filetype)
    print("Micrometers per pixel of the dataset: " + str(project_specs.dataset_mpp))
    print("Detection results will be generated to path: " + project_specs.outputpath)

    if not os.path.exists(project_specs.outputpath):
        os.makedirs(project_specs.outputpath)

    if mode == "detection":

        print("###################")
        print("## run detection ##")
        print("###################")

        ## load trained model and predict nuc locations
        NucDetectNet = cnn_functions.load_trained_model(
            project_specs.projectpath, "cnnmodel"
        )
        cnn_functions.predict_with_model(NucDetectNet, project_specs)

    elif mode == "adapt":

        print("###########################")
        print("## run domain adaptation ##")
        print("###########################")

        ## tune the trained  model with new data
        cnn_functions.domain_adapt(project_specs)

    elif mode == "train_baseline":

        print("##########################")
        print("## train baseline model ##")
        print("##########################")

        ## train baseline model
        cnn_functions.train_baseline(project_specs)


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
