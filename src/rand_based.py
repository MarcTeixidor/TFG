import numpy as np

class Random_Recommender():

    def __init__(self, config, data):
        self.data = data
        self.topk = int(config['topk'])

    def make_recommendation(self):
        song_index_rec = np.random.randint(high=100, size=1)

        return song_index_rec

    def make_k_recommendations(self):
        song_index_rec = np.random.randint(100, size=self.topk)

        return song_index_rec

    def get_data_len(self):
        
        return self.data.shape