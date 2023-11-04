#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ycbloader.py: Dataset for handling YCB data

@author: Dhruv Parikh
@brief: A lean implementation of the dataset class for our use case. Avoiding dependency on folder structure
@date: 4th November 2023
"""

import logging
import torch
from torchvision import transforms
import yaml
from torch.utils.data.dataset import Dataset
import os
from os.path import join as opj
import glob
import numpy as np
from typing import Dict, Any

def handlePath(root, isSynthetic, classId=5, mode=0)-> Dict[str, Any]:
    """
    Handles the Path of different datatypes. Synthetic Data has different folder structure while video has different.
    
    Parameters
    ----------
    root : str
        Root Dir of the data
    isSynthetic: bool
        Set to True for Synthetic, False for Real (video).
    classId:
        id of class that are there in pose_gt of video data
    mode : bool
            Set to 0 for train, 1 for test.
    """
    data = {}
    if isSynthetic:
        if mode == 0:
            root = opj(root,"validation_data_blender_DR")
        else:
            root = opj(root, "train_data_blender_DR")
        data['rgbA'] = sorted(glob.glob(opj(root, '*rgbA.png')))
        data['rgbB'] = sorted(glob.glob(opj(root, '*rgbB.png')))
        data['depthA'] = sorted(glob.glob(opj(root, '*depthA.png')))
        data['depthB'] = sorted(glob.glob(opj(root, '*depthB.png')))
        data['npzFiles'] = sorted(glob.glob(opj(root, '*meta.npz')))
    else:
        data['rgb'] = sorted(glob.glob(opj(root, 'color/*.png')))
        data['depth'] = sorted(glob.glob(opj(root, 'depth/*.png')))
        data['pose_gt'] = sorted(glob.glob(opj(root, f'pose_gt/{classId}/*.txt')))
        data['K'] = [opj(root, "cam_K.txt")]
    
    #Sanity Checks for empty keys
    for key, file_list in data.items():
        if len(file_list)==0:
            raise Exception(f"Files for {key} not found! \n Finding in Path: {root}")

    return data #We can handle maxLen here, but then I want to consider all data in shuffle and then cut off :p 

class dataloader(Dataset):
    """
    Dataloader for the YCB Data
    """

    def __init__(self, root : str, mode : bool, datatype : bool, config : str, datatransform : transforms.Compose,labeltransform : transforms.Compose, classId = 5, maxLen = None)->None:
        """
        Dataloader class for YCB Data

        Parameters
        ----------
        root : str
            Root Dir of the data
        mode : bool
            Set to 0 for train, 1 for test.
        datatype : bool
            Set to True for Synthetic, False for Real (video).
        config : str
            Location of the config file.
        datatransform : transforms.Compose
            The transformations to apply to the data.
        labeltransform : transforms.Compose
            The transformations to apply to the labels.
        classId: int
            Id of class as stated in documentation
        maxLen : int
            Maximum Length of the dataset
        """
        super().__init__()
        files = handlePath(root, datatype, classId, mode)
        for key, file_list in files.items():
            for file_path in file_list[:15]:
                print(file_path)


    def __len__(self):
        return 0

    def __getitem__(self, idx):
        pass



def main()->None:    
    root = r"data\mustard_bottle"
    datatype = 1
    classId = 5
    mode = 0
    config = ""
    datatransform = None
    labeltransform = None
    maxLen = None
    loader = dataloader(root, mode, datatype, config, datatransform, labeltransform, classId, maxLen)

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    main()
