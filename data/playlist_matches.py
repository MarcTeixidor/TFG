import json
import os

data_path = "Million Playlist Dataset/spotify_million_playlist_dataset/data/"
percent = 0.8

# List with all mpd data files path
files = os.listdir(data_path)

f = open('matches.json')
matches = json.load(f)

i=0
artists = set()
overall = 0
i=0
for f in files:
    mpd = open(data_path + f)
    mpd = json.load(mpd)['playlists']

    for playlist in mpd:
        print(i, overall)
        tracks = playlist['tracks']
        counter = 0
        for track in tracks:
            if track['artist_name'] in matches:
                counter += 1
        if counter >= int(percent*len(tracks)):
            overall+=1
        i+=1

print("There are " + overall + " playlists out of " + i + " with " + percent*100 + "% of matched artists.")