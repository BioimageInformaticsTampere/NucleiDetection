#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Created on Tue Sep 01 09:25:45 2020

Constants used in the NucleiDetection project

@author: hhakk
"""

from pathlib import Path

# CONSTANTS USED IN PROJECT

DOMAIN_ADAPTATION_THRESHOLD1 = 0.8
DOMAIN_ADAPTATION_THRESHOLD2 = 0.5

CNNMODEL = "cnnmodel"
DAMODEL = "DA-model"

# LOGGING DIRECTORY

LOGDIR = DATA = Path(__file__).parents[2].joinpath("logs")

# DATA PATHS

DATA = Path(__file__).parents[2].joinpath("data")
DATA_CONF = Path(DATA).joinpath("CONF") # Placeholder for confidence images for DA
DATA_HE = Path(DATA).joinpath("HE")
DATA_ANNOTATIONS = Path(DATA).joinpath("annotations")

# REPORT PATHS

REPORTS = Path(__file__).parents[2].joinpath("reports")
REPORTS_FIGURES = Path(REPORTS).joinpath("figures")

# MODEL PATHS

CONVNET = Path(__file__).parents[2].joinpath("models", "convnets")
TRAINED_MODELS = Path(__file__).parents[2].joinpath("models", "trained_models")

# CONFIG FILE

CONFIGFILE = Path(__file__).parents[2].joinpath("config.ini")
DEFAULT_CONFIG = {
    "model_path": TRAINED_MODELS,
    # "detection" / "adapt"
    "MODE": "detection",
    # Micormeters per pixel in model
    "model_mpp": 1.0,
    "batchoverlap": True,
    "masktype": "ball",
    "mask_element_size": 2,
    "block_size": 64,
    "batch_size": 128,
    "nb_epoch": 1,
    "network_ds": 1,
    "imagepath": DATA_HE,
    "confpath": DATA_CONF,
    "image_filetype": "tif",
    # Ground truth path
    "gtpath": Path(DATA).joinpath("gt"),
    "gt_filetype": "tif",
    "dataset_mpp": 0.5,
    "outputpath": REPORTS,
    "mask_identifier_suffix": "_mask",
    "mask_identifier_prefix": "",
    # supported: confidence,coordinates,visualisation"
    "output_mode": "confidence,coordinates,visualisation",
}
