#!/usr/bin/env python3

from pathlib import Path

# CONSTANTS USED IN PROJECT

# LOGGING DIRECTORY

LOGDIR = DATA = Path(__file__).parents[1].joinpath("logs")

# DATA PATHS

DATA = Path(__file__).parents[1].joinpath("data")
DATA_HE = Path(DATA).joinpath("HE")
DATA_ANNOTATIONS = Path(DATA).joinpath("annotations")

# MODEL PATHS

CONVNET = Path(__file__).parents[1].joinpath("models", "convnets")
TRAINED_MODELS = Path(__file__).parents[1].joinpath("models", "trained_models")
