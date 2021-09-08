import implicit
import scipy

class MF_Recommender():

    def __init__(self, config, data):
        self.data = data
        self.topk = config['topk']

        # Initialize and train the model
        self.model = self.get_model()


    def get_model(self):
        model = implicit.als.AlternatingLeastSquares(factors=64)
        model = model.fit(self.data)

        return model

    def make_recommendation(self, uid):
        rec = self.model.recommend(uid, self.data, N=1)

        return rec

    def make_k_recommendations(self, uid):
        recs = self.model.recommend(uid, self.data, N=self.topk)

        return recs

    def get_data_len(self):
        
        return self.data.shape