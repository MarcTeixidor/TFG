import scipy
import numpy as np

class Random_Recommender():

    def __init__(self, config, data):
        self.data = data
        self.topk = config['topk']

    def make_recommendation(self):
        rng = np.random.default_rng()
        song_index_rec = rng.integers(low=0, high=len(self.data[0])-1, size=1)

        return song_index_rec

    def make_k_recommendations(self):
        rng = np.random.default_rng()
        song_index_rec = rng.integers(low=0, high=len(self.data[0])-1, size=self.topk)

        return song_index_rec

    def get_data_len(self):
        
        return self.data.shape