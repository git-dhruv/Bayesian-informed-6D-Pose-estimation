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
import cv2
from PIL import Image

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
        datalen = len(data['rgbA'])
    else:
        data['rgb'] = sorted(glob.glob(opj(root, 'color/*.png')))
        data['depth'] = sorted(glob.glob(opj(root, 'depth/*.png')))
        data['pose_gt'] = sorted(glob.glob(opj(root, f'pose_gt/{classId}/*.txt')))
        data['K'] = [opj(root, "cam_K.txt")]
        datalen = len(data['rgb'])
    
    #Sanity Checks for empty keys
    for key, file_list in data.items():
        if len(file_list)==0:
            raise Exception(f"Files for {key} not found! \n Finding in Path: {root}")
        
    return data, datalen #We can handle maxLen here, but then I want to consider all data in shuffle and then cut off :p 

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
        self.root = root
        self.isSynthetic = datatype 
        self.classId = classId
        self.mode = mode
        self.config = config
        self.datatransform = datatransform
        self.labeltransform = labeltransform
        #Get all the files required
        self.files, datalen = handlePath(root, datatype, classId, mode)
        print(maxLen)
        if maxLen is None:
            maxLen = datalen
        self.maxLen = maxLen
        
    def __len__(self):
        return self.maxLen

    def __getitem__(self, idx):
        ### Config, datatransform, labeltransform, files, ply file 
        if self.isSynthetic:
            return self.getSynthData(idx)
        return self.getRealData(idx)

    ### Workhorse of the code begin from below ###
    def getSynthData(self, idx):        
        rgbA = cv2.imread(self.files['rgbA'][idx], cv2.IMREAD_UNCHANGED)
        depthA = np.int64(cv2.imread(self.files['depthA'][idx], cv2.IMREAD_UNCHANGED))
        rgbB = cv2.imread(self.files['rgbA'][idx], cv2.IMREAD_UNCHANGED)
        depthB = np.int64(cv2.imread(self.files['depthA'][idx], cv2.IMREAD_UNCHANGED))
        meta = np.load(self.files['npzFiles'])
        C_H_A = None #Pose of Object A in Camera frame - A and B is the worst naming convention
        C_H_B = None 
        for file in meta.files:
            if 'A' in file:
                C_H_A = meta[file]
            if 'B' in file:
                C_H_B = meta[file]
        if C_H_A is None or C_H_B is None:
            raise Exception (f"Can't find ground truth pose for num: {idx}")
        
        rgbA = self.datatransform(rgbA); rgbB = self.datatransform(rgbB)
        depthA = self.datatransform(depthA);depthB = self.datatransform(depthB)
        Velocity = C_H_B[:3,-1] - C_H_A[:3,-1]
        LieAlgebra = C_H_B[:3,:3]@C_H_A[:3,:3].T
        C_H_A = self.labeltransform(C_H_A);C_H_B = self.labeltransform(C_H_B) 

        return rgbA, rgbB, depthA, depthB, C_H_A, C_H_B
        

    def getRealData(self, idx):
        """
        Dataloader's job on real data is to not supply the previous image, but to only supply the current image.
        On real data we need to render synthetic image using openGL render, which requires a lot more arguments, 
        and thus is avoided in this method to be invoked by a shared call of get_item 
        """
        
        # rgb = cv2.imread(self.files['rgb'][idx], cv2.IMREAD_UNCHANGED)
        rgb = np.array(Image.open(self.files['rgb'][idx]))
        depth = (cv2.imread(self.files['depth'][idx], cv2.IMREAD_UNCHANGED)).astype(np.float64)
        gt = np.loadtxt(self.files['pose_gt'][idx])
        K = np.loadtxt(self.files['K'][0])     
        return rgb, depth, gt, K
    
    def stackRGBD(self, rgb, d):
        #This function should work on batches 
        # targetSize = (rgb.size(0), rgb.size(1) + 1, rgb.size(2), rgb.size(3))
        rgbd = torch.cat((rgb, d), dim=1)
        # assert rgbd.size() == targetSize , "stackRGBD can't give correct output"
        return rgbd
    

def main()->None:    
    root = r"data\0050"
    datatype = 0
    classId = 5
    mode = 0
    config = ""
    datatransform = transforms.ToTensor()
    labeltransform = None
    maxLen = None
    loader = dataloader(root, mode, datatype, config, datatransform, labeltransform, classId, maxLen)
    train_loader = torch.utils.data.DataLoader(loader, shuffle=True, batch_size=1)
    for data in train_loader:
        print(data)
        break


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    main()
