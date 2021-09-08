# Specify the path for imports
import scipy
import argparse
from mf import MF_Recommender
from knn import KNN_Recommender
from popularity_based import Pop_Based_Recommender
from rand_based import Random_Recommender
from neural import Neural_Recommender
import time
import os

INPUT_PATH = "/data/"
TRAIN_DATA_PATH = 'training/mat.npz'

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
        data = scipy.sparse.load_npz(os.getcwd() + INPUT_PATH + TRAIN_DATA_PATH)
        self.topk = int(config['topk'])

        if config["model"] == 'random':
            self.model = Random_Recommender(config, data) # Call method to instantiate random model
        elif config["model"] == 'popularity_based':
            self.model = Pop_Based_Recommender(config, data) # Call method to instantiate popularity based model
        elif config["model"] == 'mf':
            self.model = MF_Recommender(config, data) # Call method to instantiate MF model
        elif config["model"] == 'knn':
            self.model = KNN_Recommender(config, data) # Call method to instantiate KNN model
        elif config["model"] == 'neural':
            config['n_users'] = data.shape[0] # Each playlist is considered as a single user
            config['n_songs'] = data.shape[1]
            config['latent_dim'] = 8 # Revise this
            config['layers'] = [16,32,16,8] # Default 5 layers, revise...
            self.model = Neural_Recommender(config) # Call method to instantiate neural MF model

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
    config = parser.parse_args().__dict__

    model = RecommenderSystems(config)
    model = model.get_model()

    start_time = time.time()
    recommendations = model.make_k_recommendations()
    print(recommendations)
    end_time = time.time() - start_time
    print(end_time)