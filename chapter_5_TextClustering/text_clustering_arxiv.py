from datasets import load_dataset
import pandas as pd
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
import matplotlib.pyplot as plt
from bertopic import BERTopic
from copy import deepcopy
from bertopic.representation import KeyBERTInspired
from bertopic.representation import MaximalMarginalRelevance
from transformers import pipeline
from bertopic.representation import TextGeneration
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

#=================================
# original_topics[topic] 是很多 (word, score)
# * 把它們 unpack
# zip() 把所有第 1 個元素放一起、所有第 2 個元素放一起
# ex:zip(
#     ("robot", 0.42),
#     ("control", 0.37),
#     ("trajectory", 0.31)
# )
# list後會變成
# [
#     ("robot", "control", "trajectory", "motion", "manipulator", "system"),
#     (0.42, 0.37, 0.31, 0.28, 0.25, 0.20)
# ]
# [0] 取第一組，也就是所有 word
# [:5] 取前5個
# join() 是把很多 string 接起來 ex: robot| control| trajectory

def topic_difference(model, original_topics, nr_topics=5):
    df = pd.DataFrame(columns=["topic", "Original", "Updated"])

    for topic in range(nr_topics):
        og_words = "| ".join(list(zip(*original_topics[topic]))[0][:5])
        new_words = "| ".join(list(zip(*model.get_topic(topic)))[0][:5])
        df.loc[len(df)] = [topic, og_words, new_words] 

    return df

#=======================================

dataset = load_dataset("MaartenGr/arxiv_nlp")["train"]

print(dataset.to_pandas().head())
print(dataset.column_names)

abstracts = dataset["Abstracts"]
titles = dataset["Titles"]

embeding_model = SentenceTransformer("thenlper/gte-small")             #載入embedding模型
embeddings = embeding_model.encode(abstracts, show_progress_bar=True)  #第一步先將摘要進行嵌入

print(embeddings.shape) 

umap_model = UMAP(                                                      #進行降維
    n_components=5, min_dist=0.0, metric='cosine',                      #fit()：讓模型學習資料
                                                                        #transform()：用已學好的模型去轉換資料
                                                                        #fit_transform()：先學習，再立刻轉換同一批資料
    random_state=42
)
reduced_embeddings = umap_model.fit_transform(embeddings)

hdbscan_model = HDBSCAN(                                                #HDBSCAN是基於密度的演算法 可以自由計算分群數量
    min_cluster_size=50,
    metric='euclidean',
    cluster_selection_method='eom'                                      #eom(Excess of Mass)
                                                                        #eom = 選「最穩定、有代表性」的 cluster；leaf = 選「最細的子 cluster」。
).fit(reduced_embeddings)

clusters = hdbscan_model.labels_

print(len(set(clusters)))                                                #set()建立一個不重複的集合 看總共分了幾群



#===將embedding壓縮至2維 用圖呈現每一個cluster的分類=============================================
reduced_embeddings_2 = UMAP(
                        n_components=2,
                        min_dist=0.0,
                        metric='cosine',
                        random_state=42
).fit_transform(embeddings)

df = pd.DataFrame(reduced_embeddings_2, columns=['x', 'y'])
df['title'] = titles
df['cluster'] = [str(c) for c in clusters]

cluster_df = df.loc[df.cluster != '-1', :]
outlier_df = df.loc[df.cluster == '-1', :]

plt.scatter(outlier_df['x'], outlier_df['y'], s=10, alpha=0.05, c='grey')
plt.scatter(
    cluster_df['x'], cluster_df['y'],
    c=cluster_df['cluster'].astype(int),
    cmap='tab20b', s=2, alpha=0.6)

plt.title('HDBSCAN Clustering of ArXiv NLP Abstracts (UMAP Reduced)')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')  

plt.savefig("clusters.png", bbox_inches="tight")

#====================================================================================



#=====#前面是手動學習BERT過程 但可以省略 直接把模型架好 送到BERT==========
# umap_model = UMAP(
#     n_components=5,
#     min_dist=0.0,
#     metric="cosine",
#     random_state=42
# )

# hdbscan_model = HDBSCAN(
#     min_cluster_size=50,
#     metric="euclidean",
#     cluster_selection_method="eom"
# )

# topic_model = BERTopic(
#     embedding_model=embeding_model,
#     umap_model=umap_model,
#     hdbscan_model=hdbscan_model,
#     verbose=True
# )

# topics, probs = topic_model.fit_transform(
#     abstracts,
#     embeddings
# )


topic_model = BERTopic(                                                  
    embedding_model=embeding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    verbose=True
).fit(abstracts, embeddings)                           #需要abstracts 跟embedding是因為 透過embedding 先做cluster 
                                                       #然後從cluster裡的那些abstract做c-TF-IDF 做topic modeling

print(topic_model.get_topic_info())                    #快速獲取找到的主題描述
print(topic_model.get_topic(0))                        #查看個別主題 探索最能代表這篇文章的關鍵字
print(topic_model.find_topics("topic modeling"))       #根據關鍵字找尋特定主題文章


#===============繪圖＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
fig = topic_model.visualize_documents(                 #BERTopic 的這些 visualize_*() 圖表通常是 Plotly 互動式圖表，不是單純的靜態圖片。
                                                       #把整張互動式 Plotly 圖存成一個html網頁 可以保留互動功能
                                                       #透過 firefox topic_barchart.html 即可打開
    titles,
    reduced_embeddings = reduced_embeddings,
    hide_annotations=True,
    width=1200
)

fig.write_html("documents.html")

fig_bar = topic_model.visualize_barchart()
fig_heatmap = topic_model.visualize_heatmap(n_clusters=30)
fig_hierarchy = topic_model.visualize_hierarchy()

fig_bar.write_html("topic_barchart.html")
fig_heatmap.write_html("topic_heatmap.html")
fig_hierarchy.write_html("topic_hierarchy.html")

#＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝


original_topics = deepcopy(topic_model.topic_representations_)

representation_model = KeyBERTInspired()                            #透過比較單字和前幾名最據代表性的文件嵌入 進行cossine similarity來重新排名關鍵字
topic_model.update_topics(abstracts,
                        representation_model=representation_model
)
print(topic_difference(topic_model, original_topics))

representation_model = MaximalMarginalRelevance(diversity=0.2)      #透過最大邊際相關性Maximal marginal revelance 讓主題表達法更多樣化
topic_model.update_topics(abstracts,
                        representation_model=representation_model
)
print(topic_difference(topic_model, original_topics)) 



'''  
#===舊版flan已不可用此寫法 後續可以結合Generative LLM 提供更好的主題解釋性===

prompt = """
I have a topic that contains the following documents:
[DOCUMENTS]

The topic is described by the following keywords:
'[KEYWORDS]'.

Based on the documents and keywords, what is the topic about?
"""

generator = pipeline(
    "text-generation",
    model = "google/flan-t5-small"
)

representation_model = TextGeneration(
    generator,
    prompt = prompt,
    doc_length=50,
    tokenizer="whitespace"
)

topic_model.update_topics(abstracts, representation_model=representation_model)

print(topic_difference(topic_model, original_topics))
'''
