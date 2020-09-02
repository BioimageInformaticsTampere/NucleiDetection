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

import os
import time
import logging
from datetime import datetime
import argparse
import configparser
import json

from nucleidetection.utils import auxiliary_functions, constants
from nucleidetection.models import model


def run_nucdetect(config: configparser.ConfigParser) -> None:
    """Train or predict nuclei based on config file

    :param config: configparser.ConfigParser() object initialized with correct values
    :returns: None
    """
    logging.debug("Input path: " + config["imagepath"])
    logging.debug("Input imagetype: " + config["image_filetype"])
    logging.debug(
        "Micrometers per pixel of the dataset: " + str(config.getfloat("dataset_mpp"))
    )
    logging.debug(
        "Detection results will be generated to path: " + config["outputpath"]
    )
    logging.debug("MODE: " + config["MODE"])

    # Create directory structure here
    output_run_dir_name = config["RUNTIME"] + "_" + config["MODE"] + "_run"
    config["outputpath"] = os.path.join(constants.REPORTS, output_run_dir_name)
    config["historypath"] = os.path.join(config["outputpath"], "history")
    config["figurepath"] = os.path.join(config["outputpath"], "figures")

    if not os.path.exists(config["outputpath"]):
        os.makedirs(config["outputpath"])

    if not os.path.exists(config["historypath"]):
        os.makedirs(config["historypath"])

    if not os.path.exists(config["figurepath"]):
        os.makedirs(config["figurepath"])

    # Backup of the config to local dir
    training_configuration = {key:value for key,value in [(option, config[option]) for option in config]}
    with open(os.path.join(config["outputpath"], "training_configuration.json"), "w") as json_file:
        json.dump(training_configuration, json_file)

    if config["MODE"] == "detection":

        # load trained model and predict nuc locations
        nuc_detect_net = model.load_trained_model(config["model_path"], "cnnmodel")
        model.predict_with_model(nuc_detect_net, config)

    elif config["MODE"] == "adapt":

        # tune the trained  model with new data
        model.domain_adapt(config)


def main():

    RUNTIME = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    LOGFILE = os.path.join(
        constants.LOGDIR, RUNTIME + "_log.log"
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
    config["RUNTIME"] = RUNTIME

    time_start = time.time()

    run_nucdetect(config)

    logging.debug("Elapsed time: " + str(time.time() - time_start))


if __name__ == "__main__":
    main()
