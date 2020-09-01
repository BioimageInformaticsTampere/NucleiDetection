#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 08:29:42 2019

@author: miravalkonen
"""

import numpy as np
from skimage.util import random_noise
from skimage.color import rgb2hsv, hsv2rgb
import random

from NucleiDetection.detect import auxiliary_functions
from NucleiDetection.data import image_loader


def get_blocks_from_coordinates(
    coordinates, input_img, block_size, num_channels, imtype
):

    if num_channels == 1:
        proc_coord = np.zeros([len(coordinates), 2], dtype="int")
        blocks = np.zeros(
            [len(coordinates), block_size[0], block_size[1], 1], dtype=imtype
        )
        stepx = int((block_size[0]) / 2)
        stepy = int((block_size[1]) / 2)

        input_img = np.pad(input_img, (block_size[0], block_size[1]), "symmetric")

        blockcounter = 0
        for i in range(len(coordinates)):

            x, y = coordinates[i, :]
            x = x + block_size[0]
            y = y + block_size[1]
            if x > block_size[0] and y > block_size[1]:
                proc_coord[blockcounter, :] = coordinates[i, :]
                block = input_img[x - stepx : x + stepx, y - stepy : y + stepy]
                # plt.imshow(block)

                blocks[blockcounter, :, :, 0] = block
                blockcounter = blockcounter + 1

        imageblocks = blocks[0:blockcounter, :, :, :]
        proc_coord = proc_coord[0:blockcounter, :]

    elif num_channels == 3:

        proc_coord = np.zeros([len(coordinates), 2], dtype="int")
        blocks = np.zeros(
            [len(coordinates), block_size[0], block_size[1], num_channels], dtype=imtype
        )
        stepx = int((block_size[0]) / 2)
        stepy = int((block_size[1]) / 2)

        R = np.pad(input_img[:, :, 0], (block_size[0], block_size[1]), "symmetric")
        G = np.pad(input_img[:, :, 1], (block_size[0], block_size[1]), "symmetric")
        B = np.pad(input_img[:, :, 2], (block_size[0], block_size[1]), "symmetric")
        input_img = np.zeros(
            (
                np.shape(input_img[:, :, 0])[0] + 2 * block_size[0],
                np.shape(input_img[:, :, 0])[1] + 2 * block_size[1],
                num_channels,
            ),
            dtype=imtype,
        )
        input_img[:, :, 0] = R
        input_img[:, :, 1] = G
        input_img[:, :, 2] = B

        blockcounter = 0
        for i in range(len(coordinates)):
            x, y = coordinates[i, :]
            x = x + block_size[0]
            y = y + block_size[1]

            if x > block_size[0] and y > block_size[1]:
                proc_coord[blockcounter, :] = coordinates[i, :]
                block = input_img[x - stepx : x + stepx, y - stepy : y + stepy, :]

                blocks[blockcounter, :, :, :] = block
                blockcounter = blockcounter + 1

        imageblocks = blocks[0:blockcounter, :, :, :]
        proc_coord = proc_coord[0:blockcounter, :]

    return imageblocks, proc_coord


def shuffle_data(iminput, minput):

    inputshape = np.shape(iminput)

    shuffle_idx = random.sample(range(inputshape[0]), inputshape[0])

    shuffled_cnninput = iminput[shuffle_idx, :, :, :]
    shuffled_maskinput = minput[shuffle_idx, :, :, :]

    return shuffled_cnninput, shuffled_maskinput


def shift_hue(x, shift=0.0):

    hsv = rgb2hsv(x)
    hsv[:, :, 0] += shift
    return hsv2rgb(hsv)


def augment_data(cnninput, maskinput):

    datashape = np.shape(cnninput)
    maskshape = np.shape(maskinput)
    datatype = cnninput.dtype
    masktype = maskinput.dtype
    amount_added_data = 1
    augmented_cnninput = np.zeros(
        [amount_added_data * datashape[0], datashape[1], datashape[2], datashape[3]],
        dtype=datatype,
    )
    augmented_maskinput = np.zeros(
        [amount_added_data * maskshape[0], maskshape[1], maskshape[2], maskshape[3]],
        dtype=masktype,
    )

    mu, sigma = 0.1, 0.01
    shift_vec = np.random.normal(mu, sigma, size=[datashape[0]]).astype("float32")

    counter = 0
    for idx in range(datashape[0]):

        sample = cnninput[idx, :, :, :]
        row, col, ch = sample.shape
        mod_method = idx % 3 + 1

        if mod_method == 1:
            shifted_sample = shift_hue(sample, shift=[shift_vec[idx]])
            shifted_sample = shifted_sample.astype("float32")
            augmented_cnninput[counter, :, :, :] = shifted_sample
            augmented_maskinput[counter, :, :, :] = maskinput[idx, :, :, :]
            counter = counter + 1

        if mod_method == 2:
            noisy_sample = random_noise(sample, mode="gaussian", seed=None, clip=True)
            augmented_cnninput[counter, :, :, :] = noisy_sample
            augmented_maskinput[counter, :, :, :] = maskinput[idx, :, :, :]
            counter = counter + 1

        if mod_method == 3:
            augmented_cnninput[counter, :, :, :] = cnninput[idx, :, :, :]
            augmented_maskinput[counter, :, :, :] = maskinput[idx, :, :, :]
            counter = counter + 1

    return augmented_cnninput, augmented_maskinput


# TH2 needs to be higher than TH
def get_input_from_confidence(
    project_specs,
    block_size=128,
    threshold=0.5,
    use_multi_threshold=False,
    threshold2=0.6,
):

    inputpath = project_specs.imagepath
    confidencepath = project_specs.outputpath

    cnninput = []
    maskinput = []
    labels = []
    all_coordinates = []

    conffiles = auxiliary_functions.get_filelist(confidencepath, "tif")
    for idx in range(len(conffiles)):

        substr = conffiles[idx].split("_conf")

        conffile = confidencepath + conffiles[idx]
        imagefile = inputpath + substr[0] + "." + project_specs.image_filetype
        img = image_loader.load_image(
            imagefile, project_specs.dataset_mpp, project_specs.model_mpp
        )
        conf = image_loader.load_conf(conffile)

        coordinates = auxiliary_functions.get_coordinates_from_confidence(
            conf, threshold, "int"
        )
        # use two thresholds, one for mask, one for selecting high confidence nuc
        if use_multi_threshold:
            coordinates2 = auxiliary_functions.get_coordinates_from_confidence(
                conf, threshold2, "int"
            )

        elif not use_multi_threshold:
            coordinates2 = coordinates

        if len(coordinates2) > 0:
            mask = auxiliary_functions.generate_binary_mask_from_coordinates(
                coordinates,
                np.shape(conf),
                project_specs.masktype,
                markersize=project_specs.mask_element_size,
            )

            numchannels = 3
            imtype = "float32"
            tmpcnninput, proc_coord = get_blocks_from_coordinates(
                coordinates2, img, (block_size, block_size), numchannels, imtype
            )
            tmpmaskinput, proc_coord_m = get_blocks_from_coordinates(
                coordinates2, mask, (block_size, block_size), 1, imtype
            )

            cnninput.extend(tmpcnninput)
            maskinput.extend(tmpmaskinput)
            all_coordinates.extend(proc_coord)
            tmp_label = [substr[0]] * len(tmpmaskinput)
            labels.extend(tmp_label)

    cnninput = np.asarray(cnninput)
    maskinput = np.asarray(maskinput)

    return cnninput, maskinput, labels, all_coordinates
