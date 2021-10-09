import scipy

class Pop_Based_Recommender():

    def __init__(self, config, data):
        self.data = data
        self.topk = int(config['topk'])
        self.popular_songs = list(self.get_popular_songs())

    def get_popular_songs(self):

        popular_songs = set()
        for i in range(self.data.shape[1]):
            ocurrences = len(scipy.sparse.find(self.data[:,i])[1])
            popular_songs.update([(i,ocurrences)])

        popular_songs = sorted(popular_songs, key=lambda tup: tup[1], reverse=True)

        return popular_songs

    def make_recommendation(self, uid):
        return self.popular_songs[0]

    def make_k_recommendations(self, uid=None):

        songs_in_playlist = scipy.sparse.find(self.data[0,:])[1]

        recs = []
        for song in self.popular_songs[:self.topk*2]:
            sid = song[0]
            if sid in songs_in_playlist:
                continue
            else:
                recs.append(sid)
        
        return recs

    def get_data_len(self):
    
        return self.data.shape