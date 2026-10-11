#====Zero-Shot Classification====
#沒有已標記資料 只有標籤本身 模型決定輸入應該與候選標籤產生相關連方式

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report
from numpy import shape

def evaluate_performance(y_true, y_pred):
    report = classification_report(
        y_true, y_pred, target_names=["negative", "positive"])
    print(report)


data = load_dataset("cornell-movie-review-data/rotten_tomatoes")

# print(data)
# print(data["train"][0])  #印出第一筆資料

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

label_embeddings = model.encode(["A negative review", "A positive review"], show_progress_bar=True)
test_embeddings = model.encode(data["test"]["text"], show_progress_bar=True)

sim_matrix = cosine_similarity(test_embeddings, label_embeddings)
print(shape(sim_matrix)) #印出相似度矩陣, shape=(test_data_size, label_size)
y_pred = sim_matrix.argmax(axis=1)

evaluate_performance(data["test"]["label"], y_pred)