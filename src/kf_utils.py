#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file: kf_utils.py
@author: Dhruv Parikh
@brief: This class handles the state updates and error coords. 
"""

import numpy as np
from scipy.spatial.transform import Rotation

class HandleStates:
    def __init__(self, initialState, noises = None):        
        self.state = initialState
        print("Estimator initialized with ", self.state)

        #Populating the noise matrices - Please put in the config file later!
        self.errCov = np.eye(6,6)*1e-3 #Whatever idgaf now
        self.Q = np.diag([1e-2, 1e-2, 1e-2, 1e-3, 1e-3, 1e-3])
        self.R = np.diag([])
    
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
        pass

    def calcK(self):
        pass

    def _updateNomState(self, errstate):
        """
        'We're all consenting adults here' - Guido van Rossum
        Please treat this method as private

        To be called after each measurment and update step

        """
        self.state[:3] += errstate[:3]
        #Since we are estimating the object pose not ours!
        rotvecUpdate = Rotation.from_rotvec(errstate[3:].flatten())*Rotation.from_quat(self.state[3:])
        self.state[3:] = rotvecUpdate.as_quat()

    def fetchState(self):
        return self.state.copy() 
    
    