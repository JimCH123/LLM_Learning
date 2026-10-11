#====Tesk-Specific Model====

from datasets import load_dataset
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset
import numpy as np
from sklearn.metrics import classification_report

def evaluate_performance(y_true, y_pred):
    report = classification_report(
        y_true, y_pred, target_names=["negative", "positive"])
    print(report)


data = load_dataset("cornell-movie-review-data/rotten_tomatoes")

# print(data)
# print(data["train"][0])  #印出第一筆資料

model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"

pipe = pipeline(
    model=model_path,
    tokenizer=model_path,
    return_all_scores=True,
    top_k=None,
    device=0,  #使用GPU
)

print(pipe.model.config.id2label)     #查看模型的標籤對應index

# test = pipe("This movie is amazing!", top_k=None)
# print(test)  #印出模型對於輸入的預測結果

y_pred = []

scores = {}

for output in tqdm(pipe(KeyDataset(data["test"], "text")),    #tqdm顯示進度條, total用來顯示bar總數長度
                   total=len(data["test"])):                  #KeyDataset將dataset轉換成可迭代的物件, 只取text欄位
    for item in output:
        scores[item["label"]] = item["score"]                 #output是list, 每個item是dict, 包含label和score
                                                              #[{'label': 'positive', 'score': 0.9824380278587341}, {'label': 'neutral', 'score': 0.012347577139735222}, {'label': 'negative', 'score': 0.005214422941207886}]

    negative_score = scores["negative"]
    positive_score = scores["positive"]

    assigment = np.argmax([negative_score, positive_score])
    y_pred.append(assigment)

evaluate_performance(data["test"]["label"], y_pred)