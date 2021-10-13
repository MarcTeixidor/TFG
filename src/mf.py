import implicit
import scipy
import pickle

class MF_Recommender():

    def __init__(self, config, data):
        self.data = data
        self.topk = int(config['topk'])

        # Initialize and train the model
        self.model = self.get_model()

    def __savemodel__(self):
        with open('mf_model', 'wb') as f:
            pickle.dump(self.model, file=f)

    def get_model(self):
        model = implicit.als.AlternatingLeastSquares(factors=64)
        model.fit(self.data)

        return model

    def make_recommendation(self, uid):
        rec = self.model.recommend(uid, self.data, N=1)

        return rec

    def make_k_recommendations(self, uid):
        recs = self.model.recommend(uid, self.data, N=self.topk)

        return recs

    def get_data_len(self):
        return self.data.shape
