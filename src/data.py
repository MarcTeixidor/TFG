import torch
import random
import pandas as pd
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset
from sklear import train_test_split


class UserSongDataset(Dataset):

    def __init__(self, user_tensor, item_tensor, target_tensor):
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, uid):
        return self.user_tensor[index], self.item_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.user_tensor.size(0)

class DataGenerator(object):

    def __init__(self, uid, playlist_data, config):
        """
        uid: user id
        sparse_row: binary sparse array for songs that belong to the playlist from uid
        """

        self.playlist_data = playlist_data
        self.uids = list(range(config['n_users']))
        self.sids = list(range(config['n_songs'])) 
        self.X_train, self.y_train, self.X_test, self.y_test = self._split()


    def _split(self):

        X = [self.uids, self.sids]
        y = self.playlist_data

        return X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    def _train_loader(self, batch_size):
        
        users, songs, playlist_binary = [], [], []

        for u in range(len(self.uids):
            for s in range(len(self.sids)):
                users.append(int(u))
                songs.append(int(s))
                playlist_binary.append(int(self.playlist_data[u][s]))

        dataset = UserSongDataset(  user_tensor=torch.LongTensor(users),
                                    item_tensor=torch.LongTensor(items),+
                                    target_tensor=torch.LongTensor(playlist_binary))

        return DataLoader(dataset, batch_size=batch_size, shuffle=True)


        
        




    
