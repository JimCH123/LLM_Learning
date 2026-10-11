#====Embedding Model====

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def evaluate_performance(y_true, y_pred):
    report = classification_report(
        y_true, y_pred, target_names=["negative", "positive"])
    print(report)


data = load_dataset("cornell-movie-review-data/rotten_tomatoes")

# print(data)
# print(data["train"][0])  #印出第一筆資料

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

train_embeddings = model.encode(data["train"]["text"], show_progress_bar=True)
test_embeddings = model.encode(data["test"]["text"], show_progress_bar=True)

clf = LogisticRegression(random_state = 42)
clf.fit(train_embeddings, data["train"]["label"])

y_pred = clf.predict(test_embeddings)

evaluate_performance(data["test"]["label"], y_pred)