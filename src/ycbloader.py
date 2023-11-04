#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ycbloader.py: Dataset for handling YCB data
"""

import logging
import torch
from torchvision import transforms
import yaml
from torch.utils.data.dataset import Dataset

def handlePath(root, datatype):
    """
    Handles the Path of different datatypes. Synthetic Data has different folder structure while video has different.
    
    Parameters
    ----------
    root : str
        Root Dir of the data
    type: bool
        Set to True for Synthetic, False for Real (video).
    """

    pass

class dataloader(Dataset):
    """
    Dataloader for the YCB Data
    """

    def __init__(self, root : str, mode : bool, datatype : bool, config : str, datatransform : transforms.Compose,labeltransform : transforms.Compose, maxLen = None)->None:
        """
        Dataloader class for YCB Data

        Parameters
        ----------
        root : str
            Root Dir of the data
        mode : bool
            Set to 0 for train, 1 for test.
        datatype : bool
            Set to True for Synthetic, False for Real (video).
        config : str
            Location of the config file.
        datatransform : transforms.Compose
            The transformations to apply to the data.
        labeltransform : transforms.Compose
            The transformations to apply to the labels.
        maxLen : int
            Maximum Length of the dataset
        """
        super().__init__()

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        pass



def main()->None:    
    loader = dataloader()

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    main()
