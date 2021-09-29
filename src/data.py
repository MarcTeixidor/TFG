import torch
import random
import os
import numpy as np
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

INPUT_PATH = "/data/"
TRAIN_FOLDER_PATH = 'training/'

class PlaylistDataset(Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])


class DataGenerator(object):

    def __init__(self, playlist_data, config):
        """
        uid: user id
        sparse_row: binary sparse array for songs that belong to the playlist from uid
        """

        self.playlist_data = playlist_data
        self.uids = list(range(config['n_users']))
        self.sids = list(range(config['n_songs']))
        self.save_files = config['save_files']
        self.X_train, self.y_train, self.X_test, self.y_test = self._split()


    def _split(self):

        X = set()
        y = []

        for u in self.uids:
            for s in self.sids:
                X.update([(u,s)])
                y.append(self.playlist_data[u, s])
        
        X = list(X)

        X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2)

        if self.save_files == True:
            path = os.getcwd() + INPUT_PATH + TRAIN_FOLDER_PATH
            np.savetxt(path + 'X_train.csv', X_train, delimiter=',')
            np.savetxt(path + 'X_test.csv', X_test, delimiter=',')
            np.savetxt(path + 'y_train.csv', y_train, delimiter=',')
            np.savetxt(path + 'y_train.csv', y_test, delimiter=',')

        return X_train, X_test, y_train, y_test
