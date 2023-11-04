#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@file: main.py
@author: Dhruv Parikh
@
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

## Custom Module ##
from processDepth import depthCompletion
from ycbloader import dataloader

class Bayesian6D:
    """
    6D Pose estimation using Bayesian Framework
    """
    
    def __init__(self):
        """
        
        """
        pass


def main()->None:    
    framework = Bayesian6D()
    

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    main()
