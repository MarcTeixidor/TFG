import torch
import random
import os
import numpy as np
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

INPUT_PATH = "/data/"
TRAIN_FOLDER_PATH = 'training/'

class myDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, user):
        return self.data[user]