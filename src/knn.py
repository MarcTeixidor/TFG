from sklearn.neighbors import NearestNeighbors
import scipy

class KNN_Recommender():

    def __init__(self, config, data):
        self.data = data
        self.topk = int(config['topk'])

        # Initialize model, fit data and compute kneighbors
        self.model = self.get_model()
        self.neighbors = self.get_kneighbors()

    def __savemodel__(self):
        with open('knn_model', 'wb') as f:
            pickle.dump(self.model, file=f)

    def get_model(self):
        model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=self.topk, n_jobs=-1)
        model.fit(self.data)

        return model

    def get_kneighbors(self):
        a, idxs = self.model.kneighbors()

        return idxs

    def make_recommendation(self, uid):
        return self.neighbors[uid, 0]

    def make_k_recommendations(self, uid):
        top_k_neighbors = self.neighbors[uid, :self.topk]
        
        uid_playlist = self.data[uid].nonzero()[1]
        recommendations = []
        for i in top_k_neighbors:
            neighbor_playlist = self.data[i].nonzero()[1]
            for song in neighbor_playlist:
                if song not in uid_playlist:
                    recommendations.append(song)
                    break
        
        return recommendations

    def get_data_len(self):
        return self.data.shape
