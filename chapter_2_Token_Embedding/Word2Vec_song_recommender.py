#使用.venv313

from urllib import request
import pandas as pd
from gensim.models import Word2Vec
import numpy as np


class SongRecommender:
    """用 Word2Vec 做歌曲推薦：歌單當句子、歌曲編號當單字。
    每個步驟是一個方法，共用的物件存在 self 裡。"""

    def __init__(self):
        # 在不同步驟之間要共用的東西，先在這裡宣告
        self.playlists = None
        self.songs_df = None
        self.model = None

    # ---------- 步驟 1：下載並整理歌單資料 ----------
    def load_playlists(self):
        data = request.urlopen(
            'https://storage.googleapis.com/maps-premium/dataset/yes_complete/train.txt'
        )

        lines = data.read().decode('utf-8').split('\n')[2:]

        self.playlists = [s.rstrip().split() for s in lines if len(s.split()) > 1]
        print(self.playlists[0])

    # ---------- 步驟 2：下載歌曲資訊（編號對照歌名） ----------
    def load_songs(self):
        songs_file = request.urlopen('https://storage.googleapis.com/maps-premium/dataset/yes_complete/song_hash.txt')
        songs_file = songs_file.read().decode('utf-8').split('\n')
        songs = [s.rstrip().split('\t') for s in songs_file]
        self.songs_df = pd.DataFrame(data = songs, columns = ['id', 'title', 'artist'])
        self.songs_df = self.songs_df.set_index('id')

        print(self.songs_df)

    # ---------- 步驟 3：訓練 Word2Vec ----------
    def train_model(self):
        self.model = Word2Vec(
            self.playlists,
            vector_size=32,
            window=20,
            negative=50,
            min_count=1,
            workers=4
        )

    # ---------- 步驟 4：取得推薦歌曲 ----------
    # 注意：這裡沿用原本的 .iloc。它能得到正確結果，是因為歌曲編號是 0, 1, 2... 連續排列，
    # 「位置」剛好等於「編號」。更穩妥的寫法是 .loc（用標籤查）。
    def get_recommendation(self, song_id):
        similar_songs = np.array(
            self.model.wv.most_similar(positive=str(song_id), topn=5)
        )[:, 0]

        return self.songs_df.iloc[similar_songs]

    # ---------- 步驟 5：示範查詢 ----------
    def show_example(self, song_id=2172):
        print(self.model.wv.most_similar(positive=str(song_id)))
        print(self.songs_df.iloc[song_id])

        print(self.get_recommendation(song_id))

    # ---------- 一次跑完整個流程 ----------
    def run(self):
        self.load_playlists()
        self.load_songs()
        self.train_model()
        

if __name__ == "__main__":
    recommender = SongRecommender()
    recommender.run()
    recommender.show_example(song_id=2000)
