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
import numpy as np

class depthCompletion:
    """
    Unsuperivsed Depth completion
    Reference: 
        J. Ku, et al., "In Defense of Classical Image Processing: Fast Depth Completion on the CPU," 2018.
    """    
    def __init__(self, maxDepth, custom_kernel, full_5, full_7, full_31):
        self.max_depth = maxDepth
        if custom_kernel is None:
            #Replace this garbage with a function later
            custom_kernel = np.asarray([[0, 0, 0, 1, 0, 0, 0],
                                        [0, 0, 1, 1, 1, 0, 0],
                                        [0, 1, 1, 1, 1, 1, 0],
                                        [1, 1, 1, 1, 1, 1, 1],
                                        [0, 1, 1, 1, 1, 1, 0],
                                        [0, 0, 1, 1, 1, 0, 0],
                                        [0, 0, 0, 1, 0, 0, 0],], dtype=np.uint8)
        if full_5 is None:
            full_5 = np.ones((5, 5), np.uint8)
        if full_7 is None:
            full_7 = np.ones((7, 7), np.uint8)
        if full_31 is None:
            full_31 = np.ones((31, 31), np.uint8)

        self.custom_kernel = custom_kernel
        self.full_kernel_5 = full_5
        self.full_kernel_7 = full_7
        self.full_kernel_31 = full_31

    def fillDepth(self,inputImage):
        #Convert the mm to meters to be consistent with the paper
        img = np.float32(inputImage.copy())/1e3

        #Valid image mask - 10 cm away
        valid_depth_mask = img>0.01

        #1. Invert the Depth
        img[valid_depth_mask] = self.max_depth - img[valid_depth_mask]

        #2. Custom Kernel
        img = cv2.dilate(img, self.custom_kernel)

        #3. Small Hole Closure
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, self.full_kernel_5)
        
        #4. Small Hole Fill
        invalid_mask = img<0.01 #Remeber depth is not changed in inversion
        img[invalid_mask] = cv2.dilate(img, self.full_kernel_7)[invalid_mask]        
        
        #5. Large Hole Fill (authors loose patience at this step tbh)
        invalid_mask = img<0.01
        img[invalid_mask] = cv2.dilate(img, self.full_kernel_31)[invalid_mask]

        #6. Median+Bilateral
        img = cv2.medianBlur(img, 5)
        img = cv2.bilateralFilter(img, 9, 1.5, 2.0) #Slow but better

        #8 Depth correction
        valid_depth_mask = img>0.01
        img[valid_depth_mask] = self.max_depth - img[valid_depth_mask]

        return img*1e3

def diamondBuiscuit():
    # Diamond Kernel Maker
    size = 5  
    center = size // 2  

    diamond_kernel = np.zeros((size, size), dtype=np.uint8)

    for i in range(center + 1):
        # Starting from the center, set "1"s moving outwards to form the upper half of the diamond
        diamond_kernel[center - i, (center - i):(center + i + 1)] = 1
        # Mirror the upper half to form the bottom half of the diamond
        diamond_kernel[center + i, (center - i):(center + i + 1)] = 1
    return diamond_kernel

def main()->None:    
    #Kernels    
    custom_kernel = np.asarray(
    [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ], dtype=np.uint8)
    FULL_KERNEL_5 = np.ones((5, 5), np.uint8)
    FULL_KERNEL_7 = np.ones((7, 7), np.uint8)
    FULL_KERNEL_17 = np.ones((17, 17), np.uint8)


    img_color = cv2.imread(r"data\0050\color\000000.png", cv2.IMREAD_UNCHANGED)
    img_depth = cv2.imread(r"data\0050\depth\000000.png", cv2.IMREAD_UNCHANGED)
    ##### Depth Completion Declaration and calling ####
    processDepth = depthCompletion(5, custom_kernel, FULL_KERNEL_5, FULL_KERNEL_7, FULL_KERNEL_17)
    out = processDepth.fillDepth(img_depth)

    #This is the dataset filler
    img_depth_filled = cv2.imread(r"data\0050\depth_filled\0000000.png", cv2.IMREAD_UNCHANGED)


    plt.figure(figsize=(10, 10))  
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for displaying
    plt.title('Original Color Image')
    plt.axis('off')  
    plt.subplot(2, 2, 2)
    plt.imshow(img_depth, cmap='jet')  
    plt.title('Original Depth Image')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(out, cmap='jet')  
    plt.title('Processed Depth Image')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(img_depth_filled, cmap='jet')  
    plt.title('Depth Filled Image')
    plt.axis('off')

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    main()
