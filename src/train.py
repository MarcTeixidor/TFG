import pandas as pd
import numpy as np
import scipy
import os
from neural import Neural_Recommender
from data import DataGenerator

import torch
import torch.nn as nn
import torch.optim as optim

# Data path
INPUT_PATH = "/data/"
TRAIN_DATA_PATH = 'training/mat.npz'

# Global settings
EPOCHS = 200
BATCH_SIZE = 64

# Load data
data = scipy.sparse.load_npz(os.getcwd() + INPUT_PATH + TRAIN_DATA_PATH)
data = data[0:1000,0:1000]

# Config for the model
config = {
    'n_users': int(data.shape[0]),
    'n_songs': int(data.shape[1]),
    'latent_dim_mf': 8,
    'latent_dim_mlp': 8,
    'layers': [16,32,16,8]
}

net = Neural_Recommender(config=config)

criterion = nn.BCELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

print('-----------  1  ------------')

# DataLoader for training
sample_generator = DataGenerator(playlist_data=data, config=config)
trainset = []

print('-----------  2  ------------')

for i in range(len(sample_generator.X_train)):
    trainset.append([sample_generator.X_train[i][0], sample_generator.X_train[i][1], sample_generator.y_train[i]])

trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
#testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

print('-----------  3  ------------')

for epoch in range(EPOCHS):

    running_loss = 0.0
    for i, in_out in enumerate(trainloader, 0):
        user, song, rating = in_out
        rating = torch.FloatTensor([i for i in rating])
        rating = torch.reshape(rating, (BATCH_SIZE, 1))

        optimizer.zero_grad()

        outputs = net(user, song)
        loss = criterion(outputs, rating)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
    break

print('Finished Training')