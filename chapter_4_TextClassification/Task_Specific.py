#====Task-Specific Model====

from datasets import load_dataset
from transformers import pipeline
from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset
import numpy as np
from sklearn.metrics import classification_report


class SentimentClassifier:

    def __init__(
        self,
        dataset_name="cornell-movie-review-data/rotten_tomatoes",
        model_path="cardiffnlp/twitter-roberta-base-sentiment-latest",
        device=0
    ):
        """初始化 SentimentClassifier"""

        self.dataset_name = dataset_name
        self.model_path = model_path
        self.device = device

        self.data = None
        self.pipe = None


    def load_data(self):
        """載入資料集"""
        self.data = load_dataset(self.dataset_name)
        # print(data)
        # print(data["train"][0])  #印出第一筆資料


    def load_model(self):
        """建立 Hugging Face classification pipeline"""
        self.pipe = pipeline(
            model=self.model_path,
            tokenizer=self.model_path,
            top_k=None,
            device=self.device,  #使用GPU
        )


    def show_model_labels(self):
        """查看模型 label 與 index 的對應關係"""
        print(self.pipe.model.config.id2label)     #查看模型的標籤對應index


    def test_sentence(self, text):
        """測試單一句子"""
        test = self.pipe(text, top_k=None)
        print(test)  #印出模型對於輸入的預測結果
        return test


    def predict(self, split="test"):
        """對整個 dataset 進行預測"""
        y_pred = []

        for output in tqdm(
            self.pipe(KeyDataset(self.data[split], "text")),    #tqdm顯示進度條, total用來顯示bar總數長度
            total=len(self.data[split])                         #KeyDataset將dataset轉換成可迭代的物件, 只取text欄位
        ):

            scores = {}

            for item in output:
                scores[item["label"]] = item["score"]           #output是list, 每個item是dict, 包含label和score
                                                                #[{'label': 'positive', 'score': 0.9824380278587341}, {'label': 'neutral', 'score': 0.012347577139735222}, {'label': 'negative', 'score': 0.005214422941207886}]

            negative_score = scores["negative"]
            positive_score = scores["positive"]

            assignment = np.argmax([
                negative_score,
                positive_score
            ])
            y_pred.append(assignment)

        return y_pred


    def evaluate_performance(self, y_true, y_pred):
        """評估 classification performance"""
        report = classification_report(
            y_true,
            y_pred,
            target_names=["negative", "positive"]
        )
        print(report)


    def run(self):
        """執行完整流程"""
        #載入資料
        self.load_data()

        #載入模型
        self.load_model()

        #查看模型標籤
        self.show_model_labels()

        #測試單一句子
        self.test_sentence("This movie is amazing!")

        #預測 test dataset
        y_pred = self.predict(split="test")

        #評估模型
        self.evaluate_performance(
            self.data["test"]["label"],
            y_pred
        )


if __name__ == "__main__":
    classifier = SentimentClassifier()
    classifier.run()