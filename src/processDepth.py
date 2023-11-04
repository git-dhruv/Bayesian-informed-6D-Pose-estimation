#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
processDepth.py: Depth Completion Module

Unsupervised Depth Completion
"""

import sys
import logging

class depthCompletion:
    """
    Unsuperivsed Depth completion
    Reference: 
        J. Ku, et al., "In Defense of Classical Image Processing: Fast Depth Completion on the CPU," 2018.
    """    
    def __init__(self):
        pass


def main()->None:    
    processDepth = depthCompletion()

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    main()
