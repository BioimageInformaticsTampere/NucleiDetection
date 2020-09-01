# NucleiDetection

Supplementary material for article "Generalized fixation invariant nuclei detection through domain adaptation based deep learning" by Valkonen et al.


## Usage

Set the parameters in input argument file (more details in the main file, NucDetect.py):

EXAMPLE input argument file (input_arguments.txt):
use ':' as delimiter before and after each input argument

```
    MODE:detection:
    imagepath:/data/histoimages/:
    image_filetype:tif:
    dataset_mpp:0.5:
    outputpath:/data/output/:
    output_mode:confidence,coordinates,visualisation:
    mask_path:/data/masks/:
    mask_filetype:png:
    mask_identifier_suffix:_mask:
```


For running the NucDetect script, cd to the script path:

python NucDetect.py input_arguments.txt


## Prediction

The algorithm loads the trained CNN model from current working directory.
There should be two files:

cwd/model/cnnmodel.h5
cwd/model/cnnmodel.json

## Domain adaptation

The DA model will be saved in:

cwd/model/DA-model.h5
cwd/model/DA-model.json

## Installation

```
git clone https://github.com/BioimageInformaticsTampere/NucleiDetection
cd NucleiDetection


# Recommended: create a new virtual environment
conda create -n nucleidetection
source activate nucleidetection

pip install -e .
```



contact: valkonen.mira@gmail.com
