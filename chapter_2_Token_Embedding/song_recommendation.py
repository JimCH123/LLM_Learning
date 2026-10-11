#使用.venv313

from urllib import request
import pandas as pd
from gensim.models import Word2Vec
import numpy as np

#========================================================================
def print_recommendation(song_id):
    similar_songs = np.array(
        model.wv.most_similar(positive=str(song_id), topn=5)
    )[:, 0]

    return songs_df.iloc[similar_songs]

# ===========self-write==================
# def print_recommendation(song_id):
#     similar_song = []
#     for song, _ in model.wv.most_similar(positive=str(song_id), topn=5): 
#         similar_song.append(song)

#     return(songs_df.iloc[similar_song])
#=============================================================================


data = request.urlopen(
    'https://storage.googleapis.com/maps-premium/dataset/yes_complete/train.txt'
)

lines = data.read().decode('utf-8').split('\n')[2:]

playlists = [s.rstrip().split() for s in lines if len(s.split()) > 1]
print(playlists[0])

songs_file = request.urlopen('https://storage.googleapis.com/maps-premium/dataset/yes_complete/song_hash.txt')
songs_file = songs_file.read().decode('utf-8').split('\n')
songs = [s.rstrip().split('\t') for s in songs_file]
songs_df = pd.DataFrame(data = songs, columns = ['id', 'title', 'artist'])
songs_df = songs_df.set_index('id')

print(songs_df)

model = Word2Vec(
    playlists,
    vector_size=32,
    window=20,
    negative=50,
    min_count=1,
    workers=4
)

song_id = 2172
print(model.wv.most_similar(positive=str(song_id)))
print(songs_df.iloc[2172])


print(print_recommendation(2172))
