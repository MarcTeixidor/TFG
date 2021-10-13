from neural import Neural_Recommender
from data import myDataset

import torch
import scipy.sparse as sp
import numpy as np
import os

# Data path
INPUT_PATH = "/data/"
TRAIN_DATA_PATH = 'sparsemat/mat.npz'

# Model path
MODEL_PATH = "/models/"

# Global settings
EPOCHS = 5
BATCH_SIZE = 64

# Load data
mat = sp.load_npz(os.getcwd() + INPUT_PATH + TRAIN_DATA_PATH)
mat = mat.tocoo()
mat = torch.sparse.LongTensor(torch.LongTensor([mat.row.tolist(), mat.col.tolist()]),
                              torch.LongTensor(mat.data.astype(np.int32)))

dataset = myDataset(mat)
trainloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Config for the model
config = {
    'n_users': int(mat.shape[0]),
    'n_songs': int(mat.shape[1]),
    'latent_dim_mf': 8,
    'latent_dim_mlp': 8,
    'layers': [16,32,16,8],
    'save_files': True
}

net = Neural_Recommender(config=config)

loss_function = torch.nn.BCELoss()

optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(EPOCHS):

    running_loss = 0.0
    for i, in_out in enumerate(trainloader, 0):
        user = in_out.coalesce().indices()[0]
        song = in_out.coalesce().indices()[1]
        targets = in_out.coalesce().values().float()
        optimizer.zero_grad()

        outputs = net(user, song)
        outputs = torch.reshape(outputs, (-1,))
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 10 == 9:
            print('[%d, %5d] loss: %.3f' %
                (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

# Save trained model
torch.save(net, os.getcwd() + MODEL_PATH + 'neural_model.pth')

print('Finished training')
