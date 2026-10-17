#====Embedding Model====
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


class EmbeddingClassifier:
    """凍結的 Embedding 模型 + 邏輯回歸，做影評情緒分類。"""

    def __init__(self):
        # 在不同步驟之間要共用的東西，先在這裡宣告
        self.data = None
        self.model = None
        self.train_embeddings = None
        self.test_embeddings = None
        self.clf = None
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

    # ---------- 步驟 3：把文字編碼成向量 ----------
    def encode_data(self):
        self.train_embeddings = self.model.encode(self.data["train"]["text"], show_progress_bar=True)
        self.test_embeddings = self.model.encode(self.data["test"]["text"], show_progress_bar=True)

    # ---------- 步驟 4：訓練分類器 ----------
    def train_classifier(self):
        self.clf = LogisticRegression(random_state = 42)
        self.clf.fit(self.train_embeddings, self.data["train"]["label"])

    # ---------- 步驟 5：預測測試集 ----------
    def predict(self):
        self.y_pred = self.clf.predict(self.test_embeddings)

    # ---------- 步驟 6：評估 ----------
    def evaluate(self):
        self.evaluate_performance(self.data["test"]["label"], self.y_pred)

    # ---------- 一次跑完整個流程 ----------
    def run(self):
        self.load_data()
        self.load_embedding_model()
        self.encode_data()
        self.train_classifier()
        self.predict()
        self.evaluate()


if __name__ == "__main__":
    classifier = EmbeddingClassifier()
    classifier.run()