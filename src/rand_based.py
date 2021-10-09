import numpy as np

class Random_Recommender():

    def __init__(self, config, data):
        self.data = data
        self.topk = int(config['topk'])
        self.n_songs = int(config['n_songs'])

    def make_recommendation(self):
        song_index_rec = np.random.randint(high=self.n_songs, size=1)

        return song_index_rec

    def make_k_recommendations(self):
        song_index_rec = np.random.randint(self.n_songs, size=self.topk)

        return song_index_rec

    def get_data_len(self):
        return self.data.shape
        