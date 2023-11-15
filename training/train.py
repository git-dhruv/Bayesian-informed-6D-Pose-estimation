#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train.py: Training algorithm
"""
## System Configuration ##
import sys
import os
filedir = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.join(filedir, '..')
sys.path.append(ROOT)

## Other imports ##
import yaml
import logging
import numpy as np
import torch
from os.path import join as opj
import torch.nn as nn

## Modules ##
import models.network as network
from src import utils, ycbloader, processDepth, manipulation

import pytorch_lightning as pl
from tqdm import tqdm
#Benchmark Env
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.benchmark = True



class train(pl.LightningModule):
    """
    Update: Now wrapped with Lightning for more standard functions
    """
    
    def __init__(self, dataConfig, trainConfig, dataDir, ):
        """
        """
        self.transnorm = dataConfig['max_translation']
        self.rotnorm = dataConfig['max_rotation']
        self.logdir = opj(ROOT, 'logs')

        ## Dataset setup ##        
        isSynth = 1; maxLen = None; dataConfig = ""; classId = 5; datatype = 1        
        labeltransform= {'translation': self.transnorm, 'rotation': self.rotnorm}        
        loader = ycbloader.dataloader(dataDir,isSynth, datatype, dataConfig, None, labeltransform, classId, maxLen)


        ## Training Config ##
        epochs = trainConfig['ep']
        lr = trainConfig['lr']
        batchSize = trainConfig['batch']
        self.chkpt = None
        if len(trainConfig['pretrained'])>0:
            self.chkpt = trainConfig['pretrained']
        self.device = trainConfig['device'] #Handle exceptions


        ## Training Setup ##
        self.train_loader = torch.utils.data.DataLoader(loader, shuffle=True, batch_size=batchSize)
        self.model = network.Se3TrackNet()

        self.tLoss = nn.MSELoss()
        self.rLoss = nn.MSELoss()


    def cookInputData(self):
        #### This function should handle literally everything that is required ####
        pass

    def forward(self, img1, img2):
        t, r = self.model.forward(img1, img2)
        return t, r

    def training_step(self, batch, batch_idx):
        images, targets = batch
        img1, img2 = self.cookInputData(images)
        trans, rot = self.forward(img1, img2)
        
        tloss = self.tLoss(trans, targets[0])
        rloss = self.rLoss(rot, targets[1])

        loss =  tloss + rloss #TODO Add weights
        return loss
        
    
    # def validation_step(self, batch, batch_idx):
    #     images, targets = batch

    def on_train_epoch_end(self):
        pass

    # def on_validation_epoch_end(self):
    #     pass

    
    def train_dataloader(self):
        trainloader = torch.utils.data.DataLoader(self.train_loader, shuffle=True, batch_size=self.batchSize)
        return trainloader
    
    # def val_dataloader(self):
    #     testloader = torch.utils.data.DataLoader(self.test_loader, shuffle=False, batch_size=64)
    #     return testloader




def main()->None:    
    config = r"C:\Users\dhruv\Desktop\680Final\data\mustard_bottle\dataset_info.yml" 
    with open(config,'r') as ff:
        config = yaml.safe_load(ff)
    imagemean = np.load(r"C:\Users\dhruv\Desktop\680Final\weights\YCB_weights\mustard_bottle\mean.npy")
    imagestd = np.load(r"C:\Users\dhruv\Desktop\680Final\weights\YCB_weights\mustard_bottle\std.npy")
    modelweights = r"C:\Users\dhruv\Desktop\680Final\weights\YCB_weights\mustard_bottle\model_epoch150.pth.tar"
    transnormalize = 0.03;rotnormalize =5*np.pi/180 ; 
    meshfile = r"C:\Users\dhruv\Desktop\680Final\data\CADmodels\006_mustard_bottle\textured.ply"
    framework = train(config, imagemean, imagestd, modelweights, transnormalize, rotnormalize, meshfile)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()



"""
-------- Tracker --------
- Validation Loss left
- Using premade network
- Augmentation left
- Metrics left
- Lot of thoughts left
"""