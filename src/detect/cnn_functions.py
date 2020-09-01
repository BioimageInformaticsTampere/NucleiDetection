#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 09:28:11 2019

@author: miravalkonen
"""


import numpy as np
import os
from keras.models import model_from_json
from keras.optimizers import Adam
from skimage import io
import copy
from datetime import datetime

from NucleiDetection.detect import auxiliary_functions
from NucleiDetection.detect import image_loader
from NucleiDetection.detect import data_functions


def train_model(
    CONVnet,
    project_specs,
    cnninput,
    maskinput,
    bsize=20,
    nb_epoch=5,
    validation_data=False,
    val_input=None,
    val_masks=None,
):

    if hasattr(project_specs, "batch_size"):
        bsize = project_specs.batch_size
    if hasattr(project_specs, "nb_epoch"):
        nb_epoch = project_specs.nb_epoch

    history = CONVnet.fit(
        cnninput, maskinput, batch_size=bsize, epochs=nb_epoch, verbose=1
    )

    historyfilepath = project_specs.projectpath + "trainhistory" + os.path.sep
    if not os.path.exists(historyfilepath):
        os.makedirs(historyfilepath)

    save_trained_model(CONVnet, project_specs.projectpath, "DA-model")
    save_training_history(history, historyfilepath, "DA-model", False)

    return CONVnet


def domain_adapt(project_specs):

    TH1 = 0.8
    TH2 = 0.5
    block_size = project_specs.block_size

    NucDetectNet = load_trained_model(project_specs.projectpath, "cnnmodel")

    cnninput, maskinput, labels, coordinates = data_functions.get_input_from_confidence(
        project_specs,
        block_size=block_size,
        threshold=TH1,
        use_multi_threshold=True,
        threshold2=TH2,
    )

    cnninput, maskinput = data_functions.augment_data(cnninput, maskinput)
    cnninput, maskinput = data_functions.shuffle_data(cnninput, maskinput)

    NucDetectNet = train_model(NucDetectNet, project_specs, cnninput, maskinput)


def train_baseline(project_specs):

    return 1


def predict_with_model(CONVnet, project_specs):

    inputpath = project_specs.imagepath
    outputpath = project_specs.outputpath
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)

    filetype = project_specs.image_filetype

    imagefiles = auxiliary_functions.get_filelist(inputpath, filetype)
    for idx in range(len(imagefiles)):

        substr = imagefiles[idx].split(".")
        imagefile = inputpath + imagefiles[idx]
        img = image_loader.load_image(
            imagefile, project_specs.dataset_mpp, project_specs.model_mpp
        )

        imsize = np.shape(img)
        cnninput = np.zeros([1, imsize[0], imsize[1], imsize[2]], dtype="float32")
        cnninput[0, :, :, :] = copy.copy(img)

        prediction = CONVnet.predict(cnninput, batch_size=1, verbose=1)
        pred = (prediction[0, :, :, 0] * 255).astype("ubyte")

        for j in range(len(project_specs.output_mode)):
            outmode = project_specs.output_mode[j]
            if outmode == "confidence":
                fname = outputpath + substr[0] + "_conf.tif"
                io.imsave(fname, pred.astype("ubyte"))
            if outmode == "coordinates":
                fname = outputpath + substr[0] + "_coord.tif"

                coordinates = auxiliary_functions.get_coordinates_from_confidence(
                    pred, 0.5, "int"
                )
                mask = auxiliary_functions.generate_binary_mask_from_coordinates(
                    coordinates, [imsize[0], imsize[1]], "coordinates"
                )
                io.imsave(fname, mask.astype("ubyte"))
            if outmode == "visualisation":
                fname = outputpath + substr[0] + "_vis.tif"

                coordinates = auxiliary_functions.get_coordinates_from_confidence(
                    pred, 0.5, "int"
                )
                vis_img = auxiliary_functions.get_coordinate_overlay(
                    copy.copy(img), coordinates
                )
                io.imsave(fname, vis_img.astype("ubyte"))


def compile_model(model):

    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
    model.compile(loss="binary_crossentropy", optimizer=adam, metrics=["acc"])

    return model


def load_trained_model(path, modelname):

    # load model
    weightpath = "".join([path, modelname, ".h5"])
    jsonpath = "".join([path, modelname, ".json"])

    json_file = open(jsonpath, "r")
    loaded_model_json = json_file.read()
    json_file.close()
    CNNmodel = model_from_json(loaded_model_json)
    # load weights into model
    CNNmodel.load_weights(weightpath)
    print("Loaded model from disk")

    CNNmodel.summary()
    CNNmodel = compile_model(CNNmodel)

    return CNNmodel


def save_trained_model(CNNmodel, path, modelname):

    # SAVE MODEL
    weightpath = "".join([path, modelname, ".h5"])
    jsonpath = "".join([path, modelname, ".json"])

    model_json = CNNmodel.to_json()
    with open(jsonpath, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    CNNmodel.save_weights(weightpath)
    print("Saved model to disk")


def save_training_history(history, savepath, modelname, validation_data=False):

    # save training history
    if validation_data:
        traininghistory = {
            "acc": history.history["acc"],
            "loss": history.history["loss"],
            "val_acc": history.history["val_acc"],
            "val_loss": history.history["val_loss"],
        }
    else:
        traininghistory = {
            "acc": history.history["acc"],
            "loss": history.history["loss"],
        }

    nowsubs = str(datetime.now()).split(" ")
    nowsubs2 = nowsubs[1].split(":")
    nowsubs3 = nowsubs2[-1].split(".")
    now = "".join(
        [
            nowsubs[0],
            "_",
            nowsubs2[0],
            "-",
            nowsubs2[1],
            "-",
            nowsubs3[0],
            "-",
            nowsubs3[1],
        ]
    )
    outfilename = "".join([savepath, now, "-trainhistory.npy"])
    np.save(outfilename, traininghistory)
