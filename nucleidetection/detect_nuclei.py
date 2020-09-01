#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 08:18:26 2019

RUN NucDetect script:
    cd to the script directory
    python detect_nuclei.py input_arguments.txt > output.txt 2> errors.txt

    ## this command prints the script outputs in output.txt file and
    ## errors in errors.txt file

input_arguments.txt can contain the following input arguments:

    ARG-1 : MODE
        possible modes are:
            detection - just run nuclei detection with trained model
            adapt - fine tune the network with different dataset

    ARG-2 : imagepath
        full path to input images, path should end with /

    ARG-3 : image_filetype
        filetype of input images

    ARG-4 : dataset_mpp
        physical resolution of input images, micrometers per pixel (mpp)
        for example 40x magnification could be approximately 0.23 depending on the scanner specs

    ARG-5 : outputpath
        full output path, will be created if the path does not exist, path should end with /

    ARG-6 : output_mode
        the type of output that will be generated,
        at least one, but if multiple, use ',' as delimiter
        possible ouput modes are:
            confidence - raw output of the model as tif image
            coordinates - binary image where each nuc location corresponds to one pixel in the image
            visualisation - overlay detections over the original image

    optional-ARG-9 : mask_identifier_suffix
        mask identifier if the filename differs from input images
        image: 123.tif maks: 123_mask.tif

    optional-ARG-10 : mask_identifier_prefix
        mask identifier if the filename differs from input images
        image: 123.tif maks: mask_123.tif

EXAMPLE input argument file:
use ':' as delimiter before and after the input argument

    MODE:detection:
    imagepath:/data/histoimages/:
    image_filetype:tif:
    dataset_mpp:0.5:
    outputpath:/data/output/:
    output_mode:confidence,coordinates,visualisation:
    mask_path:/data/masks/:
    mask_filetype:png:
    mask_identifier_suffix:_mask:


@author: miravalkonen
contact: valkonen.mira@gmail.com
"""

import sys
import os
import time
import logging
from datetime import datetime
import argparse
import configparser

from nucleidetection.utils import auxiliary_functions, constants


def main():

    LOGFILE = os.path.join(
        constants.LOGDIR, datetime.now().strftime("%Y-%m-%d-%H-%M-%S_log.log")
    )
    logging.basicConfig(
        filename=LOGFILE,
        filemode="w",
        format="%(asctime)s - %(message)s",
        level=logging.DEBUG,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, default="DEFAULT", help="Configuration to choose"
    )
    args = parser.parse_args()

    config = configparser.ConfigParser()

    # We want to dynamically create the default config file to ensure
    # correct paths on different OSs

    if not os.path.exists(constants.CONFIGFILE):
        config["DEFAULT"] = constants.DEFAULT_CONFIG
        config.write(open(constants.CONFIGFILE, "w"))
    else:
        config.read(constants.CONFIGFILE)

    config = config[args.config]

    time_start = time.time()

    auxiliary_functions.run_nucdetect(config)

    logging.debug("Elapsed time: " + str(time.time() - time_start))


if __name__ == "__main__":
    main()
