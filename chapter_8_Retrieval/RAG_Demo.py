#'thenlper/gte-small'   文本嵌入模型
#../Phi-3-mini-4k-instruct-q4.gguf  LLM  生成模型
#BM25Okapi 關鍵字搜尋

import cohere
import numpy as np
import pandas as pd
from tqdm import tqdm             #顯示動態進度條的第三方庫
import faiss                      #Meta 開發的向量搜尋函式庫本身 自己算 embedding、轉 numpy、管理索引
from rank_bm25 import BM25Okapi   #關鍵字搜尋函數
import string
from sklearn.feature_extraction import _stop_words
from langchain_community.llms import LlamaCpp   #Meta開發的LLM 能在Python 環境中，用簡單的幾行程式碼載入並運行 GGUF 格式的大語言模型。
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS    #import faiss 是底層的向量搜尋引擎，FAISS（LangChain）是把它包裝好的高階工具。
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
import re
import os 
from dotenv import load_dotenv

class RAGDemo:
    EMBED_MODEL = "embed-v4.0"

    TEXTS = """
Interstellar is a 2014 epic[b] science fiction drama film[1][9] directed by Christopher Nolan, who co-wrote the screenplay with his brother, Jonathan Nolan. It features an ensemble cast led by Matthew McConaughey, Anne Hathaway, Jessica Chastain, Bill Irwin, Ellen Burstyn, and Michael Caine. Set in a dystopian future where Earth is suffering from catastrophic blight and famine, the film follows a group of astronauts who travel through space in search of a new home for humanity.

The film began development in 2006 from a treatment by producer Lynda Obst and theoretical physicist Kip Thorne, and was originally set to be directed by Steven Spielberg. Thorne served as an executive producer and scientific consultant on the film, and wrote the tie-in book The Science of Interstellar. It was Obst's final film as producer before her death. Cinematographer Hoyte van Hoytema shot it on 35 mm film in the Panavision anamorphic format and IMAX 70 mm. Filming began in late 2013 and took place in Alberta, Klaustur, and Los Angeles. Interstellar uses extensive practical and miniature effects, and the company DNEG created additional visual effects.

Interstellar premiered at the TCL Chinese Theatre on October 26, 2014, and was released in theaters in the United States on November 5, and in the United Kingdom on November 7, with Paramount Pictures distributing in the United States and Warner Bros. Pictures distributing in international markets. In the United States, it was first released on film stock, expanding to venues using digital projectors. It was a commercial success, grossing $681 million worldwide during its initial theatrical run, and $774.7 million worldwide with subsequent releases, making it the 10th-highest-grossing film of 2014. The film received generally positive reviews from critics. Among its various accolades, Interstellar was nominated for five awards at the 87th Academy Awards, winning for Best Visual Effects.
"""
    LOCAL_LLM_MODEL = "../Phi-3-mini-4k-instruct-q4.gguf"
    LOCAL_EMBED_MODEL = "thenlper/gte-small"

    LOCAT_PROMPT_TEMPLATE = """
<|user|>
Relevant information:
{context}                             

Provide a concise answer to the following question using the relevant information provided above:
{question}
<|end|>
<|assistant|>
"""
    def __init__(self):
        self.embed_model = self.EMBED_MODEL
        self.texts = self.TEXTS

        self.cohere_client = None
        self.texts_embeds = None

        self.index = None

        self.tokenized_corpus = None

        self.local_llm = None
        self.local_embed_mdoel = None
        self.local_db = None
        self.local_prompt = None

    def load_cohere(self):
        load_dotenv()
        api_key = os.getenv("COHERE_API_KEY")
        self.cohere_client = cohere.Client(api_key)

    def load_texts(self):
        # print(self.texts)
        self.texts = re.sub(r'\[[^\]]*\]', '', self.texts)
        sentences = re.split(r'(?<=[.!?])\s+', self.texts.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        # print(sentences)
        # 3. 每 3 句合成一塊，每次往後移 1 句（相鄰塊會重疊）
        window = 3
        self.texts = [
            " ".join(sentences[i:i + window])
            for i in range(max(1, len(sentences) - window + 1))
        ]
        # print(self.texts)

    def texts_embedding(self):
        self.texts_embeds = self.cohere_client.embed(
            texts = self.texts,
            model = self.embed_model,
            input_type = 'search_document',
        ).embeddings

        self.texts_embeds = np.array(self.texts_embeds)

    def build_search_index(self):
        #這個函式負責將 embeddings 加入到 FAISS 索引（index）中。
        # 一開始是先將 embeddings 轉換為 numpy 的陣列，然後用 texts_embeds.shape[1] 取得每個 embedding 的維度
        # 放進 IndexFlatL2 裡
        # print(self.texts_embeds.shape)
        self.index = faiss.IndexFlatL2(self.texts_embeds.shape[1])
        # print(index.is_trained)
        self.index.add(np.float32(self.texts_embeds))

    def dense_search(self, query, number_of_results = 5):
        query_embed = self.cohere_client.embed(                  #獲取query的嵌入向量
        texts = [query],
        model = self.embed_model,
        input_type = "search_query",
        ).embeddings[0]
        # print(query_embed)
        # print(type(query_embed))

        distances, similar_item_ids = self.index.search(        #檢索最近的鄰居
            np.float32([query_embed]),
            k = number_of_results,
        )
        # print(distances)
        # print(similar_item_ids)

        texts_np = np.array(self.texts)                         #將文本轉換為numpy以便於索引
        results = pd.DataFrame( data = 
                                {
                                    "texts" : texts_np[similar_item_ids[0]],
                                    "distances" : distances[0]
                                }    
                            )
        return results

    def texts_tokenize(self):
        self.tokenized_corpus = []
        for passage in tqdm(self.texts):
            tokenized_doc = self.bm25_tokenizer(passage)
            self.tokenized_corpus.append(tokenized_doc)

    @staticmethod
    def bm25_tokenizer(text):
        tokenized_doc = []
        for token in text.lower().split():
            token = token.strip(string.punctuation)  #punctuation(標點符號) 把token前後的標點符號拿掉
            if len(token) > 0 and token not in _stop_words.ENGLISH_STOP_WORDS:   #token 不能是英文 stop word(the a an is...)
                tokenized_doc.append(token)
        return tokenized_doc

    def keyword_and_rerank_search(self, query, num_candidates = 10, top_k = 5):
        bm25 = BM25Okapi(self.tokenized_corpus)
        print("Input question:", query)
    
        bm25_scores = bm25.get_scores(self.bm25_tokenizer(query))

        #argpartition()部份排序 並回傳索引 kth（正 從其面數, 負 從後面數）指定的第 kth 個元素進行分割。
        # 分割後，kth 位置左側的元素都 <= 該位置的元素，右側的元素都 >= 該位置的元素
        top_n = np.argpartition(bm25_scores, -num_candidates)[-num_candidates:]   #[-num_candidates:]大的都被排在後面 從後面取大的那幾個
        bm25_hits = [{'corpus_id' : idx, 'score' : bm25_scores[idx]} for idx in top_n]

        #lambda x: x['score'] 等價於 def get_score(x): return x["score"], x 是 list 裡「當下正在被拿來比較的那一筆 dictionary」
        bm25_hits = sorted(bm25_hits, key=lambda x: x['score'], reverse = True)

        print(f"Top 3 lexical search (BM25) hits")
        for hit in bm25_hits[0: top_k]:
            print(
                f"{ hit['score'] : .3f} \t {self.texts[hit['corpus_id']].replace("\n", " ") }"   #replace是把換行符號 \n 換成空格
            )

        #=======================ReRank==============================================================
        print(f"\n Top3 hits by rerank-API({len(bm25_hits)} BM25 hits re-ranked)")
        docs = [self.texts[hit['corpus_id']] for hit in bm25_hits]
        rerank_result = self.cohere_client.rerank(query = query, documents = docs, top_n = top_k, return_documents = True)   #逐步匹配query跟每一個初步篩選後的文本
        # print(rerank_result.results)
        for hit in rerank_result.results:
            # print(hit)
            print(f"{hit.relevance_score : .3f} \t {hit.document.text.replace("\n", " ")}")

    def grounded_generation(self, query, results):
        docs_dict = [{'text' : text} for text in results['texts']]
        # print(docs_dict)
        response = self.cohere_client.chat(
            message=query,
            documents=docs_dict,
        )
        return response

    def ask(self, query):
        results = self.dense_search(query)
        grounded_response = self.grounded_generation(query, results)
        return grounded_response.text


    def local_rag(self):
        self.local_llm = LlamaCpp(
                model_path = self.LOCAL_LLM_MODEL,
                n_gpu_layers = -1,
                max_tokens = 500,
                n_ctx = 2048,
                seed = 42,
                verbose=False, 
        )
        
        self.local_embed_mdoel = HuggingFaceEmbeddings(               #將文本轉換為數值表達法的嵌入模型
            model_name=self.LOCAL_EMBED_MODEL,
        )

        self.local_db = FAISS.from_texts(                         #建立本地端向量資料庫
            self.texts,
            self.local_embed_mdoel,
        )

        self.local_prompt = PromptTemplate.from_template(
            template=self.LOCAT_PROMPT_TEMPLATE,
        )

    def local_rag_generation(self, query):
        rag = RetrievalQA.from_chain_type(                #RAG流程
            llm = self.local_llm,
            retriever = self.local_db.as_retriever(search_kwargs={"k": 6}),
            chain_type = 'stuff',                         #stuff 這條鏈預設把文件填進名為 context 的變數，把問題填進 question
            chain_type_kwargs = {
                "prompt" : self.local_prompt
            },
            verbose = True
        ) 

        response = rag.invoke(
            {"query" : query}
        )
        
        return response

    def close(self):
        if self.local_llm is not None:
            self.local_llm.client.close()
            self.local_llm = None

    def run(self):
        self.load_cohere()
        self.load_texts()
        self.texts_embedding()
        self.build_search_index()

        self.texts_tokenize()

        self.local_rag()

        
    

if __name__ == "__main__":
    rag = RAGDemo()
    rag.run()

    query = "how precise was the science"
    query_2 = "How much income did Interstellar generate?"

    resposne = rag.dense_search(query=query)
    print(resposne)

    rag.keyword_and_rerank_search(query=query)

    grounded_response_2 = rag.ask(query_2)
    print(grounded_response_2)

    local_rag_resposne = rag.local_rag_generation(query_2)
    print(local_rag_resposne)
    rag.close()