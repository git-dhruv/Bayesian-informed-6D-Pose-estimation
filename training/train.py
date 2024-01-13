#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train.py: Training algorithm
@author: Dhruv Parikh
@brief: Training pipeline for the SE3Tracknet with modifications. 
@TODO: A lot but dont have the will for now
"""
## System Configuration ##
import sys
import os
filedir = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.join(filedir, '..')
sys.path.append(ROOT)
sys.path.append(os.path.join(filedir, '../src'))

## Other imports ##
import yaml
import logging
import numpy as np
import torch
from os.path import join as opj
import torch.nn as nn
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

## Modules ##
import models.network as network
from src import utils, ycbloader, manipulation

import pytorch_lightning as pl
from tqdm import tqdm
from scipy.spatial.transform import Rotation
from copy import deepcopy
import trimesh
from lie_torch import SO3

#Benchmark Env
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.benchmark = True



class train(pl.LightningModule):
    """
    Update: Now wrapped with Lightning for more standard functions
    """
    
    def __init__(self, dataConfig, trainConfig, dataDir ):
        """
        """
        super().__init__()

        self.transnorm = dataConfig['max_translation']
        self.rotnorm = dataConfig['max_rotation']*np.pi/180
        self.logdir = opj(ROOT, 'logs')

        ## Dataset setup ##        
        isSynth = 1; maxLen = 100000; dataConfig = ""; classId = 5; datatype = 1        
        labeltransform= {'translation': self.transnorm, 'rotation': self.rotnorm}        

        self.loader = ycbloader.dataloader(dataDir,datatype ,isSynth , dataConfig, None, labeltransform, classId, maxLen)
        isSynth = 1; maxLen = 100; dataConfig = ""; classId = 5; datatype = 0        
        self.val_loader = ycbloader.dataloader(dataDir, datatype,isSynth, dataConfig, None, labeltransform, classId, maxLen)


        ## Training Config ##
        epochs = trainConfig['ep']
        self.lr = trainConfig['lr']
        self.batchSize = trainConfig['batch']
        self.chkpt = None
        if len(trainConfig['pretrained'])>0:
            self.chkpt = trainConfig['pretrained']
        # self.device = trainConfig['device'] #Handle exceptions


        ## Training Setup ##
        self.train_dataset = torch.utils.data.DataLoader(self.loader, shuffle=True, batch_size=self.batchSize)
        self.val_dataset = torch.utils.data.DataLoader(self.val_loader, shuffle=True, batch_size=self.batchSize)
        
        self.model = network.Se3TrackNet()

        self.tLoss = nn.MSELoss()
        self.rLoss = nn.MSELoss()

        self.imageOps = utils.imageOps()

        self.geometricLoss = 0

        # if self.geometricLoss:
        print("Using Geometric Loss")
        self.pclUtils = utils.pointCloudUtils()
        meshfile = r"C:\Users\dhruv\Desktop\680Final\data\CADmodels\006_mustard_bottle\textured.ply"
        mesh = trimesh.load(meshfile)
        self.pointcloud = self.pclUtils.toOpen3dCloud(mesh.vertices).voxel_down_sample(voxel_size = 0.05)

        pts = deepcopy(np.asarray(self.pointcloud.points))
        self.pcl_diff = torch.tensor(np.column_stack((pts, np.ones((pts.shape[0], 1))))).cuda()
        # self.pcl_diff.requires_grad = True

        self.pcl_diff_gt = torch.tensor(np.column_stack((pts, np.ones((pts.shape[0], 1))))).cuda()
        # self.pcl_diff_gt.requires_grad = True


        ## Loggers for those dumbasses##
        self.losses = []
        self.itr = 0
        self.loss=0

        self.valLosses_geo = []
        self.valLosses_huber = []

        self.diffSO3 = SO3()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)        

        
    def getPixels_batch(self, poses, gt=0):
        if poses.dim() == 2:
            poses = poses.unsqueeze(1)
        # Transpose to (batch_size, 4, num_pts) and perform batch-wise matrix multiplication
        if gt:
            return poses.matmul(self.pcl_diff_gt.T.unsqueeze(0)).permute(0, 2, 1)
        else:
            return poses.matmul(self.pcl_diff.T.unsqueeze(0)).permute(0, 2, 1)
    def processLieAlg_batch(self, trans_pred, rot_pred, A_in_cam):
        B_in_cam = torch.zeros_like(A_in_cam)
        B_in_cam[:] = torch.eye(4)
        trans_pred = trans_pred * self.transnorm
        B_in_cam[:, :3, 3] = trans_pred + A_in_cam[:, :3, 3]
        rot_pred = rot_pred * self.rotnorm
        A2B_in_cam_rot = self.diffSO3.exp(rot_pred)
        B_in_cam[:, :3, :3] = A2B_in_cam_rot.matmul(A_in_cam[:, :3, :3])        
        return B_in_cam



    def forward(self, img1, img2):
        out = self.model.forward(img1, img2)
        t = out['trans'].double(); r = out['rot'].double()
        return t,r

    def training_step(self, batch, batch_idx):
        rgbdA, rgbdB, vDT, rlPose,C_H_A, C_H_B = batch
        trans, rot = self.forward(rgbdA.cuda().float(), rgbdB.cuda().float())        

        if self.geometricLoss==1:
            predPose = self.processLieAlg_batch(trans, rot, C_H_A)
            predicted_px = self.getPixels_batch(predPose)[:,:,:3]
            gt_px = self.getPixels_batch(C_H_B, gt=1)[:,:,:3]
            loss = self.tLoss((predicted_px/predicted_px[:,:,-1:])[:,:,:2] , (gt_px/gt_px[:,:,-1:])[:,:,:2]  )*1e3
            self.loss+=(loss); self.itr+=1
            return loss 
        else:
            # angularVel_reg = torch.norm(rot, dim=-1).mean() #Bx1     
            tloss = self.tLoss(trans, vDT)
            rloss = self.rLoss(rot, rlPose)
            
            loss =  tloss + 2*rloss #+ 1e-5*angularVel_reg  #TODO Add weights from config
            self.loss+=(loss); self.itr+=1

            return loss
    
    def validation_step(self, batch, batch_idx):
        rgbdA, rgbdB, vDT, rlPose,C_H_A, C_H_B = batch
        trans, rot = self.forward(rgbdA.cuda().float(), rgbdB.cuda().float())        
        tloss = self.tLoss(trans, vDT)
        rloss = self.rLoss(rot, rlPose)
            
        huber_loss =  tloss + 2*rloss 


        predPose = self.processLieAlg_batch(trans, rot, C_H_A)
        predicted_px = self.getPixels_batch(predPose)[:,:,:3]
        gt_px = self.getPixels_batch(C_H_B, gt=1)[:,:,:3]
        geo_loss = self.tLoss((predicted_px/predicted_px[:,:,-1:])[:,:,:2] , (gt_px/gt_px[:,:,-1:])[:,:,:2]  )*1e3

        self.valLosses_geo.append(geo_loss.cpu().detach().numpy())
        self.valLosses_huber.append(huber_loss.cpu().detach().numpy())
        return None 

        

    def on_train_epoch_end(self):
        self.losses.append(self.loss.cpu().detach().numpy()/self.itr)
        self.loss = 0
        self.itr = 0
        np.save("loss.pth",self.losses)
        torch.save(self.model.state_dict(), 'se3tracknet.pth')



    def on_validation_epoch_end(self):
        np.save("val_hub.pth",self.valLosses_huber)
        np.save("val_geo.pth",self.valLosses_geo)
        

    
    def train_dataloader(self):
        return self.train_dataset
    
    def val_dataloader(self):
        return self.val_dataset

    def configure_optimizers(self):
        return self.optimizer




def main()->None:    
    config = r"C:\Users\dhruv\Desktop\680Final\data\mustard_bottle\dataset_info.yml" 
    with open(config,'r') as ff:
        config = yaml.safe_load(ff)
    with open(r"configurations\trainconfig.yaml" ,'r') as ff:
        trainConfig = yaml.safe_load(ff)
    #  dataConfig, trainConfig, dataDir
    framework = train(config, trainConfig = trainConfig, dataDir=r"C:\Users\dhruv\Desktop\680Final\data\mustard_bottle")
    torch.set_float32_matmul_precision('medium')
    trainer = pl.Trainer(
        devices=1, 
        accelerator="gpu", 
        max_epochs=1
    )
    checkpoint = torch.load(r"C:\Users\dhruv\Desktop\680Final\weights\YCB_weights\mustard_bottle\model_epoch150.pth.tar") 
    framework.model.load_state_dict(checkpoint['state_dict'])
    # checkpoint = torch.load(r"C:\Users\dhruv\Desktop\680Final\se3tracknet.pth") 
    # framework.model.load_state_dict(checkpoint)
       
    
    trainer.fit(framework)



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
