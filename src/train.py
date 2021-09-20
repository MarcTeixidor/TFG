import pandas as pd
import numpy as np
from neural import GMF, MLP, Neural_Recommender

INPUT_PATH = "/data/"
TRAIN_DATA_PATH = 'training/mat.npz'

data = scipy.sparse.load_npz(os.getcwd() + INPUT_PATH + TRAIN_DATA_PATH)