import pandas as pd
import numpy as np
import scipy
import os
from neural import Neural_Recommender
from data import DataGenerator, PlaylistDataset

from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset

def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

# Data path
INPUT_PATH = "/data/"
TRAIN_DATA_PATH = 'training/raw_mat.npz'

# Model path
MODEL_PATH = "/models/"

# Global settings
EPOCHS = 5
BATCH_SIZE = 64
K_FOLDS = 5

# Load data
data = scipy.sparse.load_npz(os.getcwd() + INPUT_PATH + TRAIN_DATA_PATH)
data = data[0:1000,0:10000]

# Config for the model
config = {
    'n_users': int(data.shape[0]),
    'n_songs': int(data.shape[1]),
    'latent_dim_mf': 8,
    'latent_dim_mlp': 8,
    'layers': [16,32,16,8],
    'save_files': True
}

# Define k-fold
kfold = KFold(n_splits=K_FOLDS, shuffle=True)

# Define loss function
loss_function = nn.BCELoss()

print('-----------  1  ------------')

# DataLoader for training and testing
sample_generator = DataGenerator(playlist_data=data, config=config)
trainset = []
testset = []

for i in range(len(sample_generator.X_train)):
    trainset.append([sample_generator.X_train[i][0], sample_generator.X_train[i][1]])

for i in range(len(sample_generator.X_test)):
    testset.append([sample_generator.X_test[i][0], sample_generator.X_test[i][1]])

train_dataset = PlaylistDataset(trainset, sample_generator.y_train)
test_dataset = PlaylistDataset(testset, sample_generator.y_test)
dataset = ConcatDataset([train_dataset, test_dataset])

print('-----------  2  ------------')

for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):

    # Print
    print(f'FOLD {fold}')
    print('--------------------------------')

    # Sample elements randomly from a given list of ids, no replacement.
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
    
    # Define data loaders for training and testing data in this fold
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_subsampler)
    testloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, sampler=test_subsampler)

    # Init the neural network
    net = Neural_Recommender(config=config)
    net.apply(reset_weights)
    
    # Initialize optimizer
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(0, EPOCHS):

        print(f'Starting epoch {epoch+1}')

        running_loss = 0.0

        for i, in_out in enumerate(trainloader, 0):
            x, y = in_out
            user = x[0]
            song = x[1]

            optimizer.zero_grad()

            outputs = net(user, song)
            outputs = torch.reshape(outputs, (-1,))
            y = y.type(torch.FloatTensor)
            loss = loss_function(outputs, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 1000 == 999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

# Save trained model
torch.save(net, os.getcwd() + MODEL_PATH)

print('Finished Training')