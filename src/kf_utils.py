#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file: kf_utils.py
@author: Dhruv Parikh
@brief: This class handles the state updates and error coords. 
"""

import numpy as np
from scipy.spatial.transform import Rotation
from utils import lieGroup

def quaternion_to_matrix(quaternion):
    """
    Convert a quaternion to a corresponding matrix.
    
    Args:
    quaternion (np.array): A quaternion in the form [qw, qx, qy, qz].

    Returns:
    np.array: The corresponding matrix.
    """
    qw, qx, qy, qz = quaternion

    # Constructing the matrix
    matrix = np.array([[-qx, -qy, -qz],
                       [qw, -qz, qy],
                       [qz, qw, -qx],
                       [-qy, qx, qw]])

    # matrix = np.array([
    #     [qw, -qz, qy],
    #     [qz, qw, -qx],
    #     [-qy, qx, qw],
    #     [-qx, -qy, -qz]
    # ])

    # print(matrix)
    matrix = np.array([[1,0,0],[0,1,0],[0,0,1],[0,0,0]])

    return matrix

class HandleStates:
    def __init__(self, initialState, noises = None):        
        self.state = initialState
        print("Estimator initialized with ", self.state)

        #Populating the noise matrices - Please put in the config file later!
        self.errCov = np.eye(6,6) #Whatever idgaf now
        self.Q = np.diag([1e-2, 1e-2, 1e-2, 1e-3, 1e-3, 1e-3])*1e3
        self.R = np.diag([5e-2, 5e-2, 5e-2, 0.08, 0.08, 0.08, 0.08])*5e3

        # Lie group utils 
        self.lieUtils = lieGroup()
    
    def resetInitState(self, state):
        self.state = state
        print(f"System State Reset to {self.state}")

    def propogate(self, liealg):

        Fx = np.zeros((6,6))
        Fx[:3,:3] = np.eye(3,3)
        Fx[3:,3:] = Rotation.from_rotvec((liealg[3:]).flatten()).as_matrix()
        Fi = np.eye(6,6)
        P = self.errCov
        self.errCov = Fx@P@Fx.T + Fi@self.Q@Fi.T

        self._updateNomState(liealg)
    
    def measurement(self, liegrp):
        #Convert Lie Group to our state vector
        t, q = self.lieUtils.dissolveHomoTransform(liegrp)
        msment = np.concatenate((t,q))
        #First calculate the H matrix
        H = np.eye(7,7)
        Xdtheta = np.zeros((7,6))
        Xdtheta[:3,:3] = np.eye(3)
        Xdtheta[3:,3:] = quaternion_to_matrix(self.state[3:].flatten())*10
        H = H@Xdtheta

        #Kalman Gain 
        P = self.errCov
        K = P@H.T@np.linalg.inv(H@P@H.T + self.R)

        # Innovation
        innovation = np.zeros_like(self.state)
        innovation = msment - self.state
        # innovation[:3] = msment[:3] - self.state[:3]        
        innovation[3:] = Rotation.from_matrix( Rotation.from_quat(msment[3:]).as_matrix() @ (Rotation.from_quat(self.state[3:]).as_matrix()).T ).as_quat()
        errState = K@innovation

        # A more stable covariance update
        self.errCov = (np.eye(6,6)-K@H)@P@(np.eye(6,6)-K@H).T + K@(self.R)@K.T
        self._updateNomState(errState, msm=1)

    def _updateNomState(self, errstate, msm=0):
        """
        'We're all consenting adults here' - Guido van Rossum
        Please treat this method as private

        To be called after each measurment and update step

        """
        self.state[:3] += errstate[:3]
        #Since we are estimating the object pose not ours!
        if msm:
            # rotvecUpdate = Rotation.from_quat(self.state[3:])*Rotation.from_rotvec(errstate[3:].flatten()).inv()
            rotvecUpdate = Rotation.from_rotvec(errstate[3:].flatten())*Rotation.from_quat(self.state[3:])
        else:
            rotvecUpdate = Rotation.from_rotvec(errstate[3:].flatten())*Rotation.from_quat(self.state[3:])
        # print(Rotation.from_rotvec(errstate[3:].flatten()).as_rotvec())
        self.state[3:] = rotvecUpdate.as_quat()


    def fetchState(self):
        return self.state.copy() 
    
    