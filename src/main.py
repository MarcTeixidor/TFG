# Specify the path for imports
import scipy
import argparse
from mf import MF_Recommender
from knn import KNN_Recommender
from popularity_based import Pop_Based_Recommender
from rand_based import Random_Recommender
from neural import Neural_Recommender
import os
import pandas as pd

INPUT_PATH = "/data/"
DATA_PATH = 'training/mat_train.npz'

class RecommenderSystems():
    
    """
    Initialization module of the class. 
    Loads the data and takes as input config dict as follows:

            config = {
                'model': string with name of recsys chosen,
                'topk': desired number of recommendations
            }
    """
    def __init__(self, config):
        # Load the processed data in sparse matrix format
        # config = vars(config)
        data = scipy.sparse.load_npz(os.getcwd() + INPUT_PATH + DATA_PATH)
        #data = data[65000:, :]

        self.topk = int(config['topk'])
        config['n_users'] = data.shape[0] # Each playlist is considered as a single user
        config['n_songs'] = data.shape[1]

        if config["model"] == 'random':
            self.model = Random_Recommender(config, data) # Call method to instantiate random model
        elif config["model"] == 'popularity_based':
            self.model = Pop_Based_Recommender(config, data) # Call method to instantiate popularity based model
        elif config["model"] == 'mf':
            self.model = MF_Recommender(config, data) # Call method to instantiate MF model
        elif config["model"] == 'knn':
            self.model = KNN_Recommender(config, data) # Call method to instantiate KNN model
        elif config["model"] == 'neural':
            config['latent_dim'] = 8 # Revise this
            config['layers'] = [16,32,16,8] # Default 5 layers, revise...
            self.model = Neural_Recommender(config, data) # Call method to instantiate neural MF model

    """
    This function returns the desired instantiated model
    """
    def get_model(self):
        return self.model

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="String with model input:\n\trandom \n\tpopularity_based \n\tmf \n\tknn \n\t neural",required=True)
    parser.add_argument("-k", "--topk", help="Number of desired recommendations to make",required=True)
    parser.add_argument("-s", "--save", help="Save recommendations made by the model", required=False, default=True)
    config = parser.parse_args().__dict__

    model = RecommenderSystems(config)
    model = model.get_model()

    recommendations = []
    for i in range(model.get_data_len()[0]):
        if i % 10000 == 0:
            print(i)
        recs = model.make_k_recommendations(i)
        recommendations.append([i,recs])

    df = pd.DataFrame(recommendations)
    df.columns = ['playlist_number', 'recommendations']
    df.index = df['playlist_number']
    del df['playlist_number']
    df.to_csv(os.getcwd() + '/output/' + config['model'] + '_recommendations.csv')
    
    if config['save'] == True:
        model.__savemodel__()    
