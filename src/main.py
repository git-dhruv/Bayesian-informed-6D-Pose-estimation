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
import yaml
from copy import deepcopy
from scipy.spatial.transform import Rotation
import datetime, os

## Custom Module ##
from processDepth import depthCompletion
from ycbloader import dataloader


sys.path.append(r"C:\Users\dhruv\Desktop\680Final\models")

import network
from render import VispyRenderer
import trimesh
from manipulation import *
import utils
from kf_utils import HandleStates
from celluloid import Camera


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
    
    def __init__(self, config, imagemean, imagestd, modelweights, transnormalize, rotnormalize, meshfile, logFolder):
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

        self.model = network.Se3TrackNet(self.network_in_size[0])
        
        checkpoint = torch.load(modelweights)
        try:
            self.model.load_state_dict(checkpoint['state_dict'])
        except:
            self.model.load_state_dict(checkpoint)
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
        maxLen = None
        self.loadData = dataloader(root, mode, datatype, config, imagetransforms, labeltransform, classId, maxLen)

        self.transnormalize = transnormalize
        self.rotnormalize = rotnormalize


        self.depthfix = depthCompletion(30, None, None, None, None)
        self.inputModification = utils.inputImageHandler(self.network_in_size, self.objectwidth, self.K)
        self.imgOps = utils.imageOps()

        self.states = HandleStates(np.eye(4)) #Can be anything for all I care
        self.lieUtils = utils.lieGroup()
        self.firstPassCompl = 0
        

        self.predlogs = []
        fig = plt.figure()
        self.axs = fig.gca()
        self.camera = Camera(fig)

        self.logFolder = logFolder



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
        # self.visualize_depth_image(depth)

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

        self.predlogs.append(np.linalg.norm(vel)/np.linalg.norm(omega))

        return self.processPredict(pose, (vel,omega))
    
    def visualizePrediction(self, pose, rgb, wndname='Prediction' ):
        #Transform PCL using Pose
        # model = deepcopy(self.pointcloud)
        # model.transform(pose)

        pts = deepcopy(np.asarray(self.pointcloud.points))
        pts = np.column_stack((pts, np.ones((pts.shape[0],1))))
        pts = pose.dot(pts.T)


        uvs = self.imgOps.bckPrjctFromK(self.K, pts[:3,:].T)
        if wndname is not None:
            cur_bgr = cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
            for ii in range(len(uvs)):
                cv2.circle(cur_bgr,(uvs[ii,0],uvs[ii,1]),radius=1,color=(0,0,255),thickness=-1)
            cv2.imshow(wndname,cur_bgr)
            cur_rgb = cv2.cvtColor(cur_bgr,cv2.COLOR_BGR2RGB)
            self.axs.imshow(cur_rgb) #        ; plt.show()
            self.camera.snap()
            cv2.waitKey(1)
        return uvs
            
    def visualize_depth_image(self,depth_image):
        """
        Args:
        depth_image (numpy.ndarray): A depth image where depth is represented in some unit (e.g., millimeters).

        This function normalizes the depth image to 8-bit range and displays it.
        """
        # Normalize the depth image to 0-255 range for visualization
        depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
        depth_normalized = np.uint8(depth_normalized)
        cv2.imshow('Debug', depth_normalized)
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

    def logVar(self, fileName, var):
        fileName = os.path.join(self.logFolder, fileName)
        np.save(fileName, var)
    def runPipeline(self):
        self.poseErr = []
        self.reprjErr = []

        self.statePoses = []
        self.gtPoses = []

        train_loader = torch.utils.data.DataLoader(self.loadData, shuffle=False, batch_size=1)
        for idx, data in enumerate(train_loader):
            rgb, depth, gt, K = data
            if idx==0:                
                pose = gt[0].cpu().numpy()            
            pose = self.singlePass(pose, rgb.clone(), depth.clone())
            

            if idx%9==0:
                rot_noise_scale = 2*np.pi/180; translation_noise_scale = 5e-3
                rot_perturb_A = Rotation.from_rotvec((np.random.random(size=(3,)) - 0.5)*rot_noise_scale).as_quat()
                trans_perturb_A = (np.random.random(size=(3,)) - 0.5)*translation_noise_scale
                noisy_lie_algebra_A = self.lieUtils.makeHomoTransform(trans_perturb_A, rot_perturb_A)
                pose = noisy_lie_algebra_A@gt[0].cpu().numpy()     #
                self.firstPassCompl = 0
                
                # self.states.measurement(pose)
                # st = self.states.fetchState()
                # pose = self.lieUtils.makeHomoTransform(st[:3], st[3:])


            predicted_px = self.visualizePrediction(pose, rgb[0].numpy())
            self.poseErr.append(np.linalg.inv(pose)@gt[0].cpu().numpy())
            gt_px = self.visualizePrediction(gt[0].cpu().numpy(), rgb[0].numpy(), None)
            self.reprjErr.append( np.abs(predicted_px - gt_px).mean(axis=0) )

            self.statePoses.append(np.linalg.inv(pose))
            self.gtPoses.append(np.linalg.inv(gt[0]))
            


        self.logVar('reprj.npy', np.array(self.reprjErr))
        self.logVar('gtPoses.npy', np.array(self.gtPoses))
        self.logVar('statePoses.npy', np.array(self.statePoses))
        
        animation = self.camera.animate()
        animation.save(os.path.join(self.logFolder,'out.mp4'), fps=30)

        plt.figure()
        plt.plot(self.predlogs)
        plt.show()

        self.reprjErr = np.array(self.reprjErr)
        fig, axs = plt.subplots(2, 1, figsize=(15, 10))
        axs[0].plot(self.reprjErr[:,0], '.')
        axs[1].plot(self.reprjErr[:,1], '.')
        plt.show()
        
        trns = np.array([i[:3,-1] for i in self.poseErr])
        rots = np.array([Rotation.from_matrix(i[:3,:3]).as_rotvec()*180/np.pi for i in self.poseErr])

        np.save(os.path.join(self.logFolder, 'trnsErr.npy'),trns)
        np.save(os.path.join(self.logFolder, 'rotErr.npy'),rots)

        fig, axs = plt.subplots(2, 3, figsize=(15, 10))

        # Translational Errors
        axs[0, 0].hist(trns[:, 0], bins=40, color='blue')
        axs[0, 0].set_title('Translational Error in X')

        axs[0, 1].hist(trns[:, 1], bins=40, color='red')
        axs[0, 1].set_title('Translational Error in Y')

        axs[0, 2].hist(trns[:, 2], bins=40, color='green')
        axs[0, 2].set_title('Translational Error in Z')

        # Rotational Errors
        axs[1, 0].hist(rots[:, 0], bins=40, color='blue')
        axs[1, 0].set_title('Rotational Error around X')

        axs[1, 1].hist(rots[:, 1], bins=40, color='red')
        axs[1, 1].set_title('Rotational Error around Y')

        axs[1, 2].hist(rots[:, 2], bins=40, color='green')
        axs[1, 2].set_title('Rotational Error around Z')

        plt.tight_layout()
        plt.show()


        



def main()->None:    
    runstart = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    logDir = r'C:\Users\dhruv\Desktop\680Final\logs'
    folder = os.path.join(logDir, runstart)
    print("Making Log Folder at ", folder)
    os.mkdir(folder)


    config = r"C:\Users\dhruv\Desktop\680Final\data\mustard_bottle\dataset_info.yml" 
    with open(config,'r') as ff:
        config = yaml.safe_load(ff)
    imagemean = np.load(r"C:\Users\dhruv\Desktop\680Final\weights\YCB_weights\mustard_bottle\mean.npy")
    imagestd = np.load(r"C:\Users\dhruv\Desktop\680Final\weights\YCB_weights\mustard_bottle\std.npy")
    modelweights = r"C:\Users\dhruv\Desktop\680Final\weights\YCB_weights\mustard_bottle\model_epoch150.pth.tar"
    # modelweights = r"C:\Users\dhruv\Desktop\680Final\se3tracknet_good_performance.pth"
    transnormalize = 0.03;rotnormalize = 5*np.pi/180; 
    meshfile = r"C:\Users\dhruv\Desktop\680Final\data\CADmodels\006_mustard_bottle\textured.ply"
    framework = Bayesian6D(config, imagemean, imagestd, modelweights, transnormalize, rotnormalize, meshfile, folder)
    framework.runPipeline()
    

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    main()
