#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@file: main.py
@author: Dhruv Parikh
@ I made the code worse than the orignal one :)
"""



## Standard Imports ##
import sys
import logging
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
import torch.nn as nn
from scipy.spatial.transform import Rotation as R
import yaml
from copy import deepcopy

## Custom Module ##
from processDepth import depthCompletion
from ycbloader import dataloader


sys.path.append(r"C:\Users\dhruv\Desktop\680Final\iros20-6d-pose-tracking")

import se3_tracknet
from render import VispyRenderer
import trimesh
from manipulation import *
import utils
from kf_utils import HandleStates
import numpy as np

class Bayesian6D:
    """
    6D Pose estimation using Bayesian Framework (inference)
    Pupose of this class:
        - Invoke the dataloader class 
        - Get the images in loop
        - Invoke OpenGL and render the mesh image
        - Prediction from network
        - Display by backprojection 
        - Metrics are required but not a priority
    """
    
    def __init__(self, config, imagemean, imagestd, modelweights, transnormalize, rotnormalize, meshfile):
        """
        
        """
        self.network_in_size = (config['resolution'],config['resolution'])
        self.pclUtils = utils.pointCloudUtils()
        mesh = trimesh.load(meshfile)
        self.pointcloud = self.pclUtils.toOpen3dCloud(mesh.vertices).voxel_down_sample(voxel_size = 0.005)

        self.objectwidth = config['object_width']
        self.mean = imagemean 
        self.std = imagestd
        cam_cfg = config['camera']
        self.K = np.array([cam_cfg['focalX'], 0, cam_cfg['centerX'], 0, cam_cfg['focalY'], cam_cfg['centerY'], 0,0,1]).reshape(3,3)

        self.model = se3_tracknet.Se3TrackNet(self.network_in_size[0])
        
        checkpoint = torch.load(modelweights)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model = self.model.cuda()
        self.model.eval()

        self.renderer = VispyRenderer(meshfile, self.K, H=self.network_in_size[0], W=self.network_in_size[0])

        self.prev_rgb = None
        self.prev_depth = None

        imagetransforms = transforms.Compose([OffsetDepthDhruv(), NormalizeChannelsReal(imagemean, imagestd)])
        
        #### Replace these by config files ####
        root = r"data\0050"
        datatype = 0
        classId = 5
        mode = 0
        config = "" #Where do we use this config file
        labeltransform = None
        maxLen = 1000
        self.loadData = dataloader(root, mode, datatype, config, imagetransforms, labeltransform, classId, maxLen)

        self.transnormalize = transnormalize
        self.rotnormalize = rotnormalize


        self.depthfix = depthCompletion(5, None, None, None, None)
        self.inputModification = utils.inputImageHandler(self.network_in_size, self.objectwidth, self.K)
        self.imgOps = utils.imageOps()

        self.states = HandleStates(np.eye(4)) #Can be anything for all I care
        self.lieUtils = utils.lieGroup()
        self.firstPassCompl = 0
        

    def renderObject(self, CV_H_Ob):
        '''
        This function simulates the situation in openGL
        '''
        CV_H_GL = np.array([[1,0,0,0],
                            [0,-1,0,0],
                            [0,0,-1,0],
                            [0,0,0,1]])
        #Calculate Bounding Box around the Pose
        bbox = self.inputModification.compute_bbox(CV_H_Ob, self.K, self.objectwidth, scale=(1000, -1000, 1000))
        #Convert Object from OpenCV camera to GL
        ob2cam_gl = np.linalg.inv(CV_H_GL).dot(CV_H_Ob) #GL_H_CV @ CV@H_Ob = GL_H_Ob
        left = np.min(bbox[:, 1]);right = np.max(bbox[:, 1]);top = np.min(bbox[:, 0]);bottom = np.max(bbox[:, 0])
        #Update the Projection Matrix of OpenGL
        self.renderer.update_cam_mat(self.K, left, right, bottom, top)
        #Render the image
        rgb, depth = self.renderer.render_image(ob2cam_gl)
        return rgb, depth
    
    def singlePass(self, pose, rgb, depth):
        #Pull Internal state switch
        if self.firstPassCompl == 0:   
            t, q = self.lieUtils.dissolveHomoTransform(pose)                     
            self.states.resetInitState(np.concatenate((t,q)))
            self.firstPassCompl = 1

        #Current images have been cropped on the current state update vector
        rgb, depth = self.inputModification.cropImage(rgb, depth, pose, scale = (1000,1000,1000))
        

        depth = self.depthfix.fillDepth(depth) #Makes surreal difference (inactive till robust pipeline made)

        tmp = transforms.Compose([OffsetDepthDhruv(), NormalizeChannelsReal(self.mean, self.std)])
        rgb, depth,_ = tmp([rgb, depth, pose])

        #Prev images rendered
        rgb_prev, depth_prev = self.renderObject(pose)
        tmp = transforms.Compose([OffsetDepthDhruv(), NormalizeChannelsRender(self.mean, self.std)])
        rgb_prev, depth_prev,_ = tmp([rgb_prev, depth_prev, pose])
        


        #Stack the input images
        rgbd = self.loadData.stackRGBD(torch.Tensor(rgb).unsqueeze(0), torch.Tensor(depth).unsqueeze(0).unsqueeze(0)).cuda().float()
        rgbd_prev = self.loadData.stackRGBD(torch.Tensor(rgb_prev).unsqueeze(0), torch.Tensor(depth_prev).unsqueeze(0).unsqueeze(0)).cuda().float()

        #Inference time finally
        with torch.no_grad():
            twist = self.model(rgbd_prev, rgbd)

        #Inference single pass doesn't have batches
        vel = twist['trans'][0].data.cpu().numpy()
        omega = twist['rot'][0].data.cpu().numpy()

        return self.processPredict(pose, (vel,omega))
    
    def visualizePrediction(self, pose, rgb ):
        #Transform PCL using Pose
        model = deepcopy(self.pointcloud)
        model.transform(pose)

        uvs = self.imgOps.bckPrjctFromK(self.K, np.asarray(model.points))
        cur_bgr = cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
        for ii in range(len(uvs)):
            cv2.circle(cur_bgr,(uvs[ii,0],uvs[ii,1]),radius=1,color=(0,0,255),thickness=-1)
        cv2.imshow('Prediction',cur_bgr)        
        cv2.waitKey(1)
            

    def processPredict(self,A_in_cam,predB):
        '''
        # We are changing this ugly function soon

        Recover the predicted pose to the true pose
		@A_in_cam: 4x4 mat
		@predB: trans, rot, ...
		return ob pose in cam frame
		'''
        liealg = np.zeros((6,))
        liealg[:3] = predB[0]*self.transnormalize
        liealg[3:] = predB[1]*self.rotnormalize
        self.states.propogate(liealg)
        st = self.states.fetchState()
        return self.lieUtils.makeHomoTransform(st[:3], st[3:])

        B_in_cam = np.eye(4)
        trans_pred = predB[0]
        rot_pred = predB[1]
        trans_pred = trans_pred*self.transnormalize
        B_in_cam[:3,3] = trans_pred+A_in_cam[:3,3]
        rot_pred = rot_pred*self.rotnormalize
        A2B_in_cam_rot = cv2.Rodrigues(rot_pred)[0].reshape(3,3)
        B_in_cam[:3,:3] = A2B_in_cam_rot.dot(A_in_cam[:3,:3])
        return B_in_cam


    def runPipeline(self):
        train_loader = torch.utils.data.DataLoader(self.loadData, shuffle=False, batch_size=1)
        for idx, data in enumerate(train_loader):
            rgb, depth, gt, K = data
            if idx==0:                
                pose = gt[0].cpu().numpy()            
            pose = self.singlePass(pose, rgb.clone(), depth.clone())
            

            if idx%10==0:
                self.states.measurement(gt[0].cpu().numpy())
                # st = self.states.fetchState()
                # pose = self.lieUtils.makeHomoTransform(st[:3], st[3:])

            self.visualizePrediction(pose, rgb[0].numpy())



def main()->None:    
    config = r"C:\Users\dhruv\Desktop\680Final\data\mustard_bottle\dataset_info.yml" 
    with open(config,'r') as ff:
        config = yaml.safe_load(ff)
    imagemean = np.load(r"C:\Users\dhruv\Desktop\680Final\weights\YCB_weights\mustard_bottle\mean.npy")
    imagestd = np.load(r"C:\Users\dhruv\Desktop\680Final\weights\YCB_weights\mustard_bottle\std.npy")
    modelweights = r"C:\Users\dhruv\Desktop\680Final\weights\YCB_weights\mustard_bottle\model_epoch150.pth.tar"
    transnormalize = 0.03;rotnormalize =5*np.pi/180 ; 
    meshfile = r"C:\Users\dhruv\Desktop\680Final\data\CADmodels\006_mustard_bottle\textured.ply"
    framework = Bayesian6D(config, imagemean, imagestd, modelweights, transnormalize, rotnormalize, meshfile)
    framework.runPipeline()
    

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    main()
