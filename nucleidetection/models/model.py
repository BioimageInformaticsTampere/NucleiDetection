#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 09:28:11 2019

@author: miravalkonen, hhakk
"""

from datetime import datetime
import os
import copy
import configparser

import numpy as np
from tensorflow.keras.models import model_from_json
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from skimage import io

from nucleidetection.utils import auxiliary_functions, constants
from nucleidetection.data import image_loader
from nucleidetection.data import data_functions


def train_model(
    conv_net: Model,
    config: configparser.ConfigParser,
    cnninput: np.array,
    maskinput: np.array,
    bsize: int = 20,
    nb_epoch: int = 5,
) -> None:
    """Train nuclei detection model

    :param conv_net: keras model to train
    :param config: training/project config
    :param cnninput: Input images (X) for the model
    :param maskinput: Target images (Y) for the model
    :param bsize: Batch size for the model
    :param nb_epoch: Number of epochs to train

    :returns: None

    """

    bsize = config.getint("batch_size")
    nb_epoch = config.getint("nb_epoch")

    history = conv_net.fit(
        cnninput, maskinput, batch_size=bsize, epochs=nb_epoch, verbose=1
    )

    save_trained_model(conv_net, config["model_path"], constants.DAMODEL)
    save_training_history(history, config["historypath"], constants.DAMODEL)


def domain_adapt(config: configparser.ConfigParser()) -> None:
    """Train domain adaptation

    :param config: project config
    :returns: None
    """

    th1 = constants.DOMAIN_ADAPTATION_THRESHOLD1
    th2 = constants.DOMAIN_ADAPTATION_THRESHOLD2
    block_size = config.getint("block_size")

    nuc_detect_net = load_trained_model(config["model_path"], constants.CNNMODEL)

    cnninput, maskinput, labels, coordinates = data_functions.get_input_from_confidence(
        config,
        block_size=block_size,
        threshold=th1,
        use_multi_threshold=True,
        threshold2=th2,
    )

    # Perform data augmentation and shuffling
    cnninput, maskinput = data_functions.augment_data(cnninput, maskinput)
    cnninput, maskinput = data_functions.shuffle_data(cnninput, maskinput)

    train_model(nuc_detect_net, config, cnninput, maskinput)


def predict_with_model(conv_net: Model, config: configparser.ConfigParser) -> None:
    """Predict nuclei detection with trained model

    :param conv_net: keras Model instance
    :param config: project config

    :returns: None
    """

    inputpath = config["imagepath"]
    outputpath = config["figurepath"]

    if not os.path.exists(outputpath):
        os.makedirs(outputpath)

    filetype = config["image_filetype"]

    imagefiles = auxiliary_functions.get_filelist(inputpath, filetype)

    for idx in range(len(imagefiles)):

        output_file_basename = imagefiles[idx].split(".")
        imagefile = os.path.join(inputpath, imagefiles[idx])
        img = image_loader.load_image(
            imagefile, config.getfloat("dataset_mpp"), config.getfloat("model_mpp"),
        )

        imsize = np.shape(img)
        cnninput = np.zeros([1, imsize[0], imsize[1], imsize[2]], dtype="float32")
        cnninput[0, :, :, :] = copy.copy(img)

        prediction = conv_net.predict(cnninput, batch_size=1, verbose=1)

        pred = (prediction[0, :, :, 0] * 255).astype("ubyte")

        output_mode = [m.strip() for m in config.get("output_mode").split(",")]

        for j in range(len(output_mode)):

            outmode = output_mode[j]

            if outmode == "confidence":
                fname = os.path.join(outputpath, output_file_basename[0] + "_conf.tif")
                io.imsave(fname, pred.astype("ubyte"))
            if outmode == "coordinates":
                fname = os.path.join(outputpath, output_file_basename[0] + "_coord.tif")

                coordinates = auxiliary_functions.get_coordinates_from_confidence(
                    pred, 0.5, "int"
                )
                mask = auxiliary_functions.generate_binary_mask_from_coordinates(
                    coordinates, [imsize[0], imsize[1]], "coordinates"
                )
                io.imsave(fname, mask.astype("ubyte"))

            if outmode == "visualisation":
                fname = os.path.join(outputpath, output_file_basename[0] + "_vis.tif")

                coordinates = auxiliary_functions.get_coordinates_from_confidence(
                    pred, 0.5, "int"
                )
                vis_img = auxiliary_functions.get_coordinate_overlay(
                    copy.copy(img), coordinates
                )
                io.imsave(fname, vis_img.astype("ubyte"))


def compile_model(model: Model) -> Model:
    """Compile the Keras model

    :param model: keras Model
    :returns: compiled Model
    """

    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
    model.compile(loss="binary_crossentropy", optimizer=adam, metrics=["acc"])

    return model


def load_trained_model(path: str, modelname: str) -> Model:
    """
    Load pre-trained model from disk

    :param path: path to dir of model
    :param modelname: model filename

    :returns: keras Model instance
    """

    # load model
    weightpath = os.path.join(path, f"{modelname}.h5")
    jsonpath = os.path.join(path, f"{modelname}.json")

    json_file = open(jsonpath, "r")
    loaded_model_json = json_file.read()
    json_file.close()
    cnn_model = model_from_json(loaded_model_json)

    # load weights into model
    cnn_model.load_weights(weightpath)

    cnn_model.summary()
    cnn_model = compile_model(cnn_model)

    return cnn_model


def save_trained_model(cnn_model: Model, path: str, modelname: str) -> None:
    """Save trained model to disk

    :param cnn_model: model to save
    :param path: dir to save model
    :param modelname: model filename

    :returns: None
    """

    # SAVE MODEL
    weightpath = "".join([path, modelname, ".h5"])
    jsonpath = "".join([path, modelname, ".json"])

    model_json = cnn_model.to_json()
    with open(jsonpath, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    cnn_model.save_weights(weightpath)


def save_training_history(history, savepath: str, modelname: str) -> None:
    """Save Keras fit history to disk

    :param history: keras .fit history
    :param savepath: path to save history
    :param modelname: name of the trained model

    :returns: None
    """

    traininghistory = {
        "acc": history.history["acc"],
        "loss": history.history["loss"],
    }

    realtime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    outfilename = f"{modelname}_{realtime}_trainhistory.npy"
    outfilename = os.path.join(savepath, outfilename)
    np.save(outfilename, traininghistory)
