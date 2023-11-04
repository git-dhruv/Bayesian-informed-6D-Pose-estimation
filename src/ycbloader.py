#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ycbloader.py: Dataset for handling YCB data
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

def handlePath(root, isSynthetic, classId=5)-> Dict[str, Any]:
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
    """
    data = {}
    if isSynthetic:
        data['rgbA'] = sorted(glob.glob(opj(root, '*rgbA.png')))
        data['rgbB'] = sorted(glob.glob(opj(root, '*rgbB.png')))
        data['depthA'] = sorted(glob.glob(opj(root, '*depthA.png')))
        data['depthB'] = sorted(glob.glob(opj(root, '*depthB.png')))
        data['npzFiles'] = sorted(glob.glob(opj(root, '*meta.npz')))
    else:
        data['rgb'] = sorted(glob.glob(opj(root, 'color/*.png')))
        data['depth'] = sorted(glob.glob(opj(root, 'depth/*.png')))
        data['pose_gt'] = sorted(glob.glob(opj(root, f'pose_gt/{classId}/*.txt')))
        data['K'] = opj(root, "cam_K.txt")


    return data

class dataloader(Dataset):
    """
    Dataloader for the YCB Data
    """

    def __init__(self, root : str, mode : bool, datatype : bool, config : str, datatransform : transforms.Compose,labeltransform : transforms.Compose, maxLen = None)->None:
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
        maxLen : int
            Maximum Length of the dataset
        """
        super().__init__()

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        pass



def main()->None:    
    loader = dataloader()

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    main()
