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

        #Populate the Noise matrices - left to formulate

        

    def updateErrState(self):
        pass

    def _updateNomState(self, errstate):
        """
        'We're all consenting adults here' - Guido van Rossum
        Please treat this method as private

        To be called after each measurment and update step

        """
        pass

    def measurement(self):
        pass

    def propogate(self):
        pass

    def fetchState(self):
        return self.state.copy() 