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
sys.path.append(os.path.join(filedir, '../src'))

## Other imports ##
import yaml
import logging
import numpy as np
import torch
from os.path import join as opj
import torch.nn as nn
from torchvision import transforms

## Modules ##
import models.network as network
from src import utils, ycbloader, manipulation

import pytorch_lightning as pl
from tqdm import tqdm
from scipy.spatial.transform import Rotation
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
        self.rotnorm = dataConfig['max_rotation']
        self.logdir = opj(ROOT, 'logs')

        ## Dataset setup ##        
        isSynth = 1; maxLen = None; dataConfig = ""; classId = 5; datatype = 1        
        labeltransform= {'translation': self.transnorm, 'rotation': self.rotnorm}        
        self.loader = ycbloader.dataloader(dataDir,isSynth, datatype, dataConfig, None, labeltransform, classId, maxLen)


        ## Training Config ##
        epochs = trainConfig['ep']
        lr = trainConfig['lr']
        self.batchSize = trainConfig['batch']
        self.chkpt = None
        if len(trainConfig['pretrained'])>0:
            self.chkpt = trainConfig['pretrained']
        # self.device = trainConfig['device'] #Handle exceptions


        ## Training Setup ##
        self.train_loader = torch.utils.data.DataLoader(self.loader, shuffle=True, batch_size=self.batchSize)
        self.model = network.Se3TrackNet()

        self.tLoss = nn.MSELoss()
        self.rLoss = nn.MSELoss()

    def forward(self, img1, img2):
        out = self.model.forward(img1, img2)
        t = out['trans'].double(); r = out['rot'].double()
        return t,r

    def training_step(self, batch, batch_idx):
        rgbdA, rgbdB, vDT, rlPose = batch
        trans, rot = self.forward(rgbdA.cuda().float(), rgbdB.cuda().float())        
        tloss = self.tLoss(trans, vDT)
        rloss = self.rLoss(rot, rlPose)
        loss =  tloss + rloss #TODO Add weights
        return loss
        
    
    # def validation_step(self, batch, batch_idx):
    #     images, targets = batch

    def on_train_epoch_end(self):
        pass

    # def on_validation_epoch_end(self):
    #     pass

    
    def train_dataloader(self):
        return self.train_loader
    
    # def val_dataloader(self):
    #     testloader = torch.utils.data.DataLoader(self.test_loader, shuffle=False, batch_size=64)
    #     return testloader
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        return optimizer




def main()->None:    
    config = r"C:\Users\dhruv\Desktop\680Final\data\mustard_bottle\dataset_info.yml" 
    with open(config,'r') as ff:
        config = yaml.safe_load(ff)
    with open(r"configurations\trainconfig.yaml" ,'r') as ff:
        trainConfig = yaml.safe_load(ff)
    imagemean = np.load(r"C:\Users\dhruv\Desktop\680Final\weights\YCB_weights\mustard_bottle\mean.npy")
    imagestd = np.load(r"C:\Users\dhruv\Desktop\680Final\weights\YCB_weights\mustard_bottle\std.npy")
    modelweights = r"C:\Users\dhruv\Desktop\680Final\weights\YCB_weights\mustard_bottle\model_epoch150.pth.tar"
    transnormalize = 0.03;rotnormalize =5*np.pi/180 ; 
    meshfile = r"C:\Users\dhruv\Desktop\680Final\data\CADmodels\006_mustard_bottle\textured.ply"
    #  dataConfig, trainConfig, dataDir
    framework = train(config, trainConfig = trainConfig, dataDir=r"data\mustard_bottle")
    # framework.train()
    torch.set_float32_matmul_precision('medium')
    trainer = pl.Trainer(
        devices=1, 
        accelerator="gpu", 
        max_epochs=100
    )
    trainer.fit(framework)



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