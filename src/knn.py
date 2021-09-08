from sklearn.neighbors import NearestNeighbors
import scipy

class KNN_Recommender():

    def __init__(self, config, data):
        self.data = data
        self.topk = config['topk']

        # Initialize model, fit data and compute kneighbors
        self.model = self.get_model()
        self.neighbors = self.get_kneighbors()


    def get_model(self):
        model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=self.topk, n_jobs=-1)
        model = model.fit(self.data)

        return model

    def get_kneighbors(self):
        _, idxs = self.model.kneighbors()

        return idxs

    def make_recommendation(self, uid):
        
        return self.neighbors[uid, 0]

    def make_k_recommendations(self, uid):

        return self.neighbors[uid]

    def get_data_len(self):

        return self.data.shape