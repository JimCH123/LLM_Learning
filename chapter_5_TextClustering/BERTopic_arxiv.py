#.venv312
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
import openai
from bertopic.representation import OpenAI
from dotenv import load_dotenv


class ArxivBERTopicPipeline:
    """用 BERTopic 對 arXiv NLP 論文摘要做主題模型：
    嵌入 -> UMAP 降維 -> HDBSCAN 分群 -> c-TF-IDF 抽主題 -> 視覺化 -> 重新排名關鍵字。"""

    PROMPT = """
        I have a topic that contains the following documents:
        [DOCUMENTS]
        
        The topic is described by the following keywords:
        [KEYWORDS].
        
        Based on the information above, extract a short topic label in the following format:
        topic: <short topic label>
    """
    def __init__(self):
        self.dataset = None
        self.titles = None
        self.abstracts = None

        self.embedding_model = None
        self.embeddings = None

        self.umap_model = None
        self.reduced_embeddings = None

        self.hdbscan_model = None
        self.clusters = None

        self.reduced_embeddings_2 = None

        self.topic_model = None

        self.original_topics = None

   
    @staticmethod
    def topic_difference(model, original_topics, nr_topics=5):
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
        df = pd.DataFrame(columns=["topic", "Original", "Updated"])

        for topic in range(nr_topics):
            og_words = "| ".join(list(zip(*original_topics[topic]))[0][:5])
            new_words = "| ".join(list(zip(*model.get_topic(topic)))[0][:5])
            df.loc[len(df)] = [topic, og_words, new_words]

        return df

    # ---------- 步驟 1：載入資料 ----------
    def load_data(self):
        self.dataset = load_dataset("MaartenGr/arxiv_nlp")['train']
        # print(self.dataset)
        # print(self.dataset.to_pandas().head())
        # print(self.dataset.column_names)
        self.titles = self.dataset['Titles']
        self.abstracts = self.dataset['Abstracts']

    # ---------- 步驟 2：嵌入 --------------
    def embed(self):
        #載入embedding模型, `thenlper/gte-small` 是一個小而快的句子嵌入模型，每段文字會變成一個384 維的向量。

        self.embedding_model = SentenceTransformer("thenlper/gte-small")  
        self.embeddings = self.embedding_model.encode(self.abstracts, show_progress_bar = True)
        # print(self.embeddings.shape)

    # ---------- 步驟 3：UMAP 降維 ----------
    def reduce_dimensions(self):
        self.umap_model = UMAP(
            n_components=5,
            min_dist=0.0,
            metric='cosine',
            random_state = 42,  
        )
        self.reduced_embeddings = self.umap_model.fit_transform(self.embeddings)
        # print(self.reduced_embeddings.shape)

    # ---------- 步驟 4：HDBSCAN 分群 ----------
    def cluster(self):
        #HDBSCAN是基於密度的演算法 可以自由計算分群數量
        #eom(Excess of Mass) eom = 選「最穩定、有代表性」的 cluster；leaf = 選「最細的子 cluster」。    
        self.hdbscan_model = HDBSCAN(
            min_cluster_size=50,
            metric='euclidean',
            cluster_selection_method='eom',
        ).fit(self.reduced_embeddings)

        self.clusters = self.hdbscan_model.labels_
        # print(len(set(self.clusters)))   

    # ---------- 步驟 5：壓成 2 維並畫出每個 cluster --------
    def plot_clusters(self):
        self.reduced_embeddings_2 = UMAP(
            n_components=2,
            min_dist=0.0,
            metric='cosine',
            random_state=42,
        ).fit_transform(self.embeddings)

        df = pd.DataFrame(self.reduced_embeddings_2, columns=['x', 'y'])
        df['title'] = self.titles
        df['cluster'] = [str(c) for c in self.clusters]

        cluster_df = df.loc[df.cluster != '-1', :]
        outlier_df = df.loc[df.cluster == "-1", :]

        plt.scatter(
            cluster_df['x'], cluster_df['y'],
            c=cluster_df['cluster'].astype(int),
            cmap='tab20b', s=2, alpha=0.6,
        )
        plt.scatter(outlier_df['x'], outlier_df['y'], s=10, alpha=0.05, c='grey')

        plt.title('HDBSCAN Clustering of ArXiv NLP Abstracts (UMAP Reduced)')
        plt.xlabel('UMAP Dimension 1')
        plt.ylabel('UMAP Dimension 2')  
        plt.savefig("clusters.png", bbox_inches="tight")

    # ---------- 步驟 6：建立並訓練 BERTopic ----------
    def build_topic_model(self):
        """前面是手動學習BERT過程 但可以省略 直接把模型架好 送到BERT"""
        #需要abstracts 跟embedding是因為 透過embedding 先做cluster
        #然後從cluster裡的那些abstract做c-TF-IDF 做topic modeling
        self.topic_model = BERTopic(
            embedding_model = self.embedding_model,
            umap_model = self.umap_model,
            hdbscan_model = self.hdbscan_model,
            verbose = True,
        ).fit(self.abstracts, self.embeddings)

    # ---------- 步驟 7：查看主題 ----------------
    def inspect_topics(self):
            print(self.topic_model.get_topic_info())                    #快速獲取找到的主題描述
            print(self.topic_model.get_topic(0))                        #查看個別主題 探索最能代表這篇文章的關鍵字
            print(self.topic_model.find_topics("topic modeling"))       #根據關鍵字找尋特定主題文章

    # ---------- 步驟 8：繪圖並存成 html ----------
    def visualize(self):
        #BERTopic 的這些 visualize_*() 圖表通常是 Plotly 互動式圖表，不是單純的靜態圖片。
        #把整張互動式 Plotly 圖存成一個html網頁 可以保留互動功能
        #透過 firefox topic_barchart.html 即可打開
        fig = self.topic_model.visualize_documents(
            docs = self.titles,
            reduced_embeddings = self.reduced_embeddings_2,
            hide_annotations = True,
            width = 1200,
        )
        fig.write_html("documents.html")
        
        fig_bar = self.topic_model.visualize_barchart()
        fig_heatmap = self.topic_model.visualize_heatmap(n_clusters=30)
        fig_hierarchy = self.topic_model.visualize_hierarchy()

        fig_bar.write_html("topic_barchart.html")
        fig_heatmap.write_html("topic_heatmap.html")
        fig_hierarchy.write_html("topic_hierarchy.html")

    # ---------- 步驟 9：先存下原本的主題（之後比較用） ----------
    def save_original_topics(self):
        # 一定要在 update_topics 之前做，否則存到的就已經是改過的版本
        self.original_topics = deepcopy(self.topic_model.topic_representations_)

    # ---------- 步驟 10：用 KeyBERTInspired 重新排名關鍵字 ----------
    def rerank_with_keybert(self):
        representation_model = KeyBERTInspired()
        self.topic_model.update_topics(
            self.abstracts,
            representation_model=representation_model
        )
        print(self.topic_difference(self.topic_model, self.original_topics))

    # ---------- 步驟 11：用 MMR 讓關鍵字更多樣化 ----------
    def diversify_with_mmr(self):
        representation_model = MaximalMarginalRelevance(diversity=0.2)
        self.topic_model.update_topics(
            self.abstracts,
            representation_model=representation_model,
        )
        print(self.topic_difference(self.topic_model, self.original_topics))


    def generative_llm(self):
        load_dotenv()
        client = openai.OpenAI()
        representation_model = OpenAI(
            client,
            model="gpt-4o-mini",
            prompt=self.PROMPT,
            exponential_backoff=True,
        )
        self.topic_model.update_topics(
            self.abstracts,
            representation_model=representation_model,
        )
        print(self.topic_difference(self.topic_model, self.original_topics))
        


    def run(self):
        self.load_data()

        self.embed()
        self.reduce_dimensions()
        self.cluster()
        # self.plot_clusters()

        self.build_topic_model()
        # self.inspect_topics()
        # self.visualize()

        self.save_original_topics()
        self.rerank_with_keybert()
        self.diversify_with_mmr()
        # self.generative_llm()


if __name__ == "__main__":
    arxiv_bert = ArxivBERTopicPipeline()
    arxiv_bert.run()

