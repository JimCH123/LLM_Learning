#====Zero-Shot Classification====
#沒有已標記資料 只有標籤本身 模型決定輸入應該與候選標籤產生相關連方式

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report
from numpy import shape

class ZeroShotClassifier:
    """零樣本分類：把「標籤的文字描述」和「影評」編碼到同一個向量空間，
    比較餘弦相似度，取最近的標籤。沒有任何訓練，也不需要標註資料。
    每個步驟是一個方法，共用的物件存在 self 裡。"""

    def __init__(self):
        self.data = None
        self.model = None
        self.label_embeddings = None
        self.test_embeddings = None
        self.sim_matrix = None
        self.y_pred = None

    # ---------- 評估函式 ----------
    # 用 staticmethod：函式裡不需要 self，只依賴傳進來的兩個參數。
    @staticmethod
    def evaluate_performance(y_true, y_pred):
        report = classification_report(
            y_true, y_pred, target_names=["negative", "positive"])
        print(report)

    # ---------- 步驟 1：載入資料 ----------
    def load_data(self):
        self.data = load_dataset("cornell-movie-review-data/rotten_tomatoes")

        # print(self.data)
        # print(self.data["train"][0])  #印出第一筆資料

    # ---------- 步驟 2：載入 Embedding 模型 ----------
    def load_embedding_model(self):
        self.model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    # ---------- 步驟 3：把標籤描述與影評都編碼成向量 ----------
    def encode_data(self):
        self.label_embeddings = self.model.encode(["A negative review", "A positive review"], show_progress_bar=True)
        self.test_embeddings = self.model.encode(self.data["test"]["text"], show_progress_bar=True)

    # ---------- 步驟 4：計算相似度，取最近的標籤 ----------
    def predict(self):
        self.sim_matrix = cosine_similarity(self.test_embeddings, self.label_embeddings)
        print(shape(self.sim_matrix)) #印出相似度矩陣, shape=(test_data_size, label_size)
        self.y_pred = self.sim_matrix.argmax(axis=1)

    # ---------- 步驟 5：評估 ----------
    def evaluate(self):
        self.evaluate_performance(self.data["test"]["label"], self.y_pred)

    # ---------- 一次跑完整個流程 ----------
    def run(self):
        self.load_data()
        self.load_embedding_model()
        self.encode_data()
        self.predict()
        self.evaluate()


if __name__ == "__main__":
    classifier = ZeroShotClassifier()
    classifier.run()