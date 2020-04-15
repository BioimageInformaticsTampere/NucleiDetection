#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 08:18:26 2019

RUN NucDetect script:
    cd to the script directory
    python NucDetect.py input_arguments.txt > output.txt 2> errors.txt
    
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

def main():
    import sys
    import os
    import time
    from nuc_detect_modules import auxiliary_functions 
    
    time_start = time.time()
    #change working directory
    pathsplit = os.path.split(os.path.abspath(sys.argv[0]))
    os.chdir(pathsplit[0])
  
    print('############################################################')
    print('############################################################')
    
    print('Set working directory..: ' + os.getcwd())  
    print('Runnin script...: ' + sys.argv[0] + '\n')
   
    with open(sys.argv[1]) as f:
        arguments = f.readlines()
    
    auxiliary_functions.run_nucdetect(arguments)
    
    print('Elapsed time..: '+ str(time.time()-time_start))

if __name__ == '__main__':
    main()