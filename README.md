# NucleiDetection

Based on the tool introduced in the article:

> "Generalized fixation invariant nuclei detection through domain adaptation based deep learning" (Valkonen et al.)

Refactored by Hannu Hakkola (hannu.hakkola@tuni.fi)

This code is licenced under GPL licence. For more information, see `LICENCE` file

## Usage

Set the parameters in input argument file (more details in the main file, NucDetect.py):

EXAMPLE input argument configuration (in config.ini, created at first run of the script):

```
[NewConfiguration]
MODE = detection
imagepath = /data/histoimages/
image_filetype = tif
dataset_mpp = 0.5
outputpath = /data/output/
output_mode = confidence,coordinates,visualisation
mask_path = /data/masks/
mask_filetype = png
mask_identifier_suffix = _mask
```


For running the `detect_nuclei` script with above configuration, cd to the script path:

`python detect_nuclei.py --config NewConfiguration`

## Prediction

The algorithm loads the trained CNN model from the project directory
There should be two files:

```
./data/models/trained_models/cnnmodel.h5
./data/models/trained_models/cnnmodel.json
```

## Domain adaptation

The DA model will be saved in:

```
./data/models/trained_models/DA-model.h5
./data/models/trained_models/DA-model.json
```

## Installation

```
git clone https://github.com/BioimageInformaticsTampere/NucleiDetection
cd NucleiDetection


# Recommended: create a new virtual environment
conda create -n nucleidetection
source activate nucleidetection

pip install -r requirements.txt
```



## Contact:
* [Mira](mailto:valkonen.mira@gmail.com)
* [Hannu](mailto:hannu.hakkola@tuni.fi)
