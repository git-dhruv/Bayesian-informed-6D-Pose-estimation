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

from utils import lieGroup
from scipy.spatial.transform import Rotation
import manipulation

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
            depthB = "depthB.png"
        else:
            depthB = "depthB_fake.png"
            root = opj(root, "train_data_blender_DR")
        data['rgbA'] = sorted(glob.glob(opj(root, '*rgbA.png')))
        data['rgbB'] = sorted(glob.glob(opj(root, '*rgbB.png')))
        data['depthA'] = sorted(glob.glob(opj(root, '*depthA.png')))
        data['depthB'] = sorted(glob.glob(opj(root, f'*{depthB}')))
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
        labeltransform : Dictionary {'translation', 'rotation'}
            The transformations to apply to the labels. These are normalizers
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
        if maxLen is None:
            maxLen = datalen
        self.maxLen = maxLen

        self.mean = np.load(r"C:\Users\dhruv\Desktop\680Final\weights\YCB_weights\mustard_bottle\mean.npy")
        self.std = np.load(r"C:\Users\dhruv\Desktop\680Final\weights\YCB_weights\mustard_bottle\std.npy")

        
    def __len__(self):
        return self.maxLen

    def __getitem__(self, idx):
        """
        * This architecture assumes that dataloader's job is to only provide raw images. 
        * Note that we can't alter data before cropping. This is because, the cropping works based on 3D pose and we alter 3D cluster in augmentation. 
        * The current solution would be to load raw images, and let a utils class handle both cropping and augmentation.
        * We still however do label transform since I don't want to route 2 poses to the utils everytime.          
        """ 
        if self.isSynthetic:
            return self.getSynthData(idx)
        return self.getRealData(idx)

    ### Workhorse of the code begin from below ###
    def getSynthData(self, idx):        
        # I hate this Image open as much as you do
        rgbA = np.array(Image.open(self.files['rgbA'][idx]))
        depthA = np.float64(cv2.imread(self.files['depthA'][idx], cv2.IMREAD_UNCHANGED))
        rgbB = np.array(Image.open(self.files['rgbB'][idx]))
        depthB = np.float64(cv2.imread(self.files['depthB'][idx], cv2.IMREAD_UNCHANGED))

        


        meta = np.load(self.files['npzFiles'][idx])
        C_H_A = None #Pose of Object A in Camera frame - A and B is the worst naming convention
        C_H_B = None 
        for file in meta.files:
            if 'A' in file:
                C_H_A = meta[file]
            if 'B' in file:
                C_H_B = meta[file]
        if C_H_A is None or C_H_B is None:
            raise Exception (f"Can't find ground truth pose for num: {idx}")

        #Lord forgive me for the sin I am above to do
        rgbdA, rgbdB = self.cookInputData(rgbA, rgbB, depthA, depthB, C_H_A, C_H_B)

        vDT = C_H_B[:3,-1] - C_H_A[:3,-1]
        rlPose = C_H_B[:3,:3]@C_H_A[:3,:3].T ### Derivation: C_H_r @ C_H_A = C_H_B; C_H_r = C_H_B @ (C_H_A).T
         
        vDT = vDT/self.labeltransform['translation']
        rlPose = lieGroup().constructValidRotationMatrix(rlPose) #Ensure we get a valid rotation matrix 
        rlPose = Rotation.from_matrix(rlPose).as_rotvec()/self.labeltransform['rotation']

        return rgbdA, rgbdB, vDT, rlPose, C_H_A, C_H_B
        

    def getRealData(self, idx):
        """
        Dataloader's job on real data is to not supply the previous image, but to only supply the current image.
        On real data we need to render synthetic image using openGL render, which requires a lot more arguments, 
        and thus is avoided in this method to be invoked by a shared call of get_item 
        """
        
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
    def cookInputData(self, rgbA, rgbB, depthA, depthB, C_H_A, C_H_B):
        #### This function should handle literally everything that is required ####

        ## Add perturbation homogenous matrix ##
        rot_noise_scale = 5*np.pi/180; translation_noise_scale = 5e-2
        rot_perturb_A = Rotation.from_rotvec((np.random.random(size=(3,)) - 0.5)*rot_noise_scale).as_quat()
        trans_perturb_A = (np.random.random(size=(3,)) - 0.5)*translation_noise_scale
        noisy_lie_algebra_A = lieGroup().makeHomoTransform(trans_perturb_A, rot_perturb_A)
        rot_perturb_B = Rotation.from_rotvec((np.random.random(size=(3,)) - 0.5)*rot_noise_scale).as_quat()
        trans_perturb_B = (np.random.random(size=(3,)) - 0.5)*translation_noise_scale
        noisy_lie_algebra_B = lieGroup().makeHomoTransform(trans_perturb_B, rot_perturb_B)



        ## - We need do OffseDepth for both rendering plus real
        tmp = transforms.Compose([manipulation.OffsetDepthDhruv(), manipulation.NormalizeChannelsRender(self.mean, self.std)])
        rgbA, depthA,_ = tmp([rgbA, depthA, noisy_lie_algebra_A@C_H_A])
        tmp = transforms.Compose([manipulation.OffsetDepthDhruv(), manipulation.NormalizeChannelsReal(self.mean, self.std)])
        rgbB, depthB,_ = tmp([rgbB, depthB, noisy_lie_algebra_B@C_H_B])
        ## Return the StackRGB
        rgbd = self.stackRGBD(torch.Tensor(rgbA).unsqueeze(0), torch.Tensor(depthA).unsqueeze(0).unsqueeze(0))
        rgbd_prev = self.stackRGBD(torch.Tensor(rgbB).unsqueeze(0), torch.Tensor(depthB).unsqueeze(0).unsqueeze(0))
        return rgbd.squeeze(0), rgbd_prev.squeeze(0)


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
    train_loader = torch.utils.data.DataLoader(loader, shuffle=True, batch_size=1)
    for data in train_loader:
        print(data)
        break


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    main()
