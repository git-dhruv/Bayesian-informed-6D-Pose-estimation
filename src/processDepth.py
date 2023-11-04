#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
processDepth.py: Depth Completion Module

Unsupervised Depth Completion
"""

import sys
import logging
import cv2
import matplotlib.pyplot as plt

class depthCompletion:
    """
    Unsuperivsed Depth completion
    Reference: 
        J. Ku, et al., "In Defense of Classical Image Processing: Fast Depth Completion on the CPU," 2018.
    """    
    def __init__(self):
        pass

    def fillDepth(self,img):
        print(img.max())
        plt.imshow(img)


def main()->None:    
    img = cv2.imread(r"..\data\0050\depth\000000.png", cv2.IMREAD_UNCHANGED)
    processDepth = depthCompletion()
    processDepth.fillDepth(img)
    plt.show(block=False)
    plt.figure()
    img = cv2.imread(r"..\data\0050\depth_filled\0000000.png", cv2.IMREAD_UNCHANGED)
    processDepth.fillDepth(img)
    plt.show()

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    main()
