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

#=====================密集檢索 Dense Retrival===========================
def search(cohere_client, model, query, index, texts, number_of_results = 5):
    query_embed = cohere_client.embed(                  #獲取query的嵌入向量
        texts = [query],
        model = model,
        input_type = "search_query",
    ).embeddings[0]
    # print(query_embed)
    # print(type(query_embed))

    distances, similar_item_ids = index.search(        #檢索最近的鄰居
        np.float32([query_embed]),
        k = number_of_results,
    )
    # print(distances)
    # print(similar_item_ids)

    texts_np = np.array(texts)                         #將文本轉換為numpy以便於索引
    results = pd.DataFrame( data = 
                            {
                                "texts" : texts_np[similar_item_ids[0]],
                                "distances" : distances[0]
                            }    
                        )
    
    return results


def build_search_index(texts_embeds):
    #這個函式負責將 embeddings 加入到 FAISS 索引（index）中。
    # 一開始是先將 embeddings 轉換為 numpy 的陣列，然後用 texts_embeds.shape[1] 取得每個 embedding 的維度
    # 放進 IndexFlatL2 裡

    index = faiss.IndexFlatL2(texts_embeds.shape[1])
    # print(index.is_trained)
    index.add(np.float32(texts_embeds))
    return index


def texts_embedding(cohere_client, texts, model):
    response = cohere_client.embed(       #獲取文本的embedding 向量
        texts = texts,
        input_type = "search_document",
        model = model,
    ).embeddings
    # print(response)

    texts_embeds = np.array(response)
    # print(texts_embeds)
    # print(texts_embeds.shape)
    # print(type(texts_embeds))
    # print(texts_embeds.dtype)
    return texts_embeds


def load_texts():
    text = """
        Interstellar is a 2014 epic[b] science fiction drama film[1][9] directed by Christopher Nolan, who co-wrote the screenplay with his brother, Jonathan Nolan. It features an ensemble cast led by Matthew McConaughey, Anne Hathaway, Jessica Chastain, Bill Irwin, Ellen Burstyn, and Michael Caine. Set in a dystopian future where Earth is suffering from catastrophic blight and famine, the film follows a group of astronauts who travel through space in search of a new home for humanity.

        The film began development in 2006 from a treatment by producer Lynda Obst and theoretical physicist Kip Thorne, and was originally set to be directed by Steven Spielberg. Thorne served as an executive producer and scientific consultant on the film, and wrote the tie-in book The Science of Interstellar. It was Obst's final film as producer before her death. Cinematographer Hoyte van Hoytema shot it on 35 mm film in the Panavision anamorphic format and IMAX 70 mm. Filming began in late 2013 and took place in Alberta, Klaustur, and Los Angeles. Interstellar uses extensive practical and miniature effects, and the company DNEG created additional visual effects.

        Interstellar premiered at the TCL Chinese Theatre on October 26, 2014, and was released in theaters in the United States on November 5, and in the United Kingdom on November 7, with Paramount Pictures distributing in the United States and Warner Bros. Pictures distributing in international markets. In the United States, it was first released on film stock, expanding to venues using digital projectors. It was a commercial success, grossing $681 million worldwide during its initial theatrical run, and $774.7 million worldwide with subsequent releases, making it the 10th-highest-grossing film of 2014. The film received generally positive reviews from critics. Among its various accolades, Interstellar was nominated for five awards at the 87th Academy Awards, winning for Best Visual Effects.
    """

    texts = text.split(".")                #將文本分割成句子
    # print(texts)
    texts = [t.strip(" \n") for t in texts] #strip()（去除前後字元：移除字串開頭和結尾指定的字元（預設是空白或換行符號））

    '''
    #============Claude 處理文本方式===================================
    text = re.sub(r'\[[^\]]*\]', '', text)

    # 2. 切句：只在「句號 + 空白」處切，且排除縮寫 "Bros."
    #    $774.7 的小數點後面沒有空白，所以不會被切斷
    sentences = re.split(r'(?<!Bros\.)(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    # 3. 每 3 句合成一塊，每次往後移 1 句（相鄰塊會重疊）
    window = 3
    texts = [
        " ".join(sentences[i:i + window])
        for i in range(max(1, len(sentences) - window + 1))
    ]
    #==================================================================

    '''
    return texts


def load_cohere(api_key):
    cohere_client = cohere.Client(api_key)  
    return cohere_client

#=======================================================================

#把文章整理成「乾淨的 token 清單」
def bm25_tokenizer(text):
    tokenized_doc = []
    for token in text.lower().split():
        token = token.strip(string.punctuation)  #punctuation(標點符號) 把token前後的標點符號拿掉
        if len(token) > 0 and token not in _stop_words.ENGLISH_STOP_WORDS:   #token 不能是英文 stop word(the a an is...)
            tokenized_doc.append(token)
    return tokenized_doc


def texts_tokenize(texts):
    
    tokenized_corpus = []
    for passage in tqdm(texts):
        tokenized_doc = bm25_tokenizer(passage)
        tokenized_corpus.append(tokenized_doc)
    return tokenized_corpus


def keyword_and_rerank_search(query, bm25, texts, cohere_client, num_candidates = 15, top_k = 5):
        print("Input question:", query)

        bm25_scores = bm25.get_scores(bm25_tokenizer(query))

        #argpartition()部份排序 並回傳索引 kth（正 從其面數, 負 從後面數）指定的第 kth 個元素進行分割。
        # 分割後，kth 位置左側的元素都 <= 該位置的元素，右側的元素都 >= 該位置的元素
        top_n = np.argpartition(bm25_scores, -num_candidates)[-num_candidates:]   #[-num_candidates:]大的都被排在後面 從後面取大的那幾個
        bm25_hits = [{'corpus_id' : idx, 'score' : bm25_scores[idx]} for idx in top_n]

        #lambda x: x['score'] 等價於 def get_score(x): return x["score"], x 是 list 裡「當下正在被拿來比較的那一筆 dictionary」
        bm25_hits = sorted(bm25_hits, key=lambda x: x['score'], reverse = True)

        print(f"Top 3 lexical search (BM25) hits")
        for hit in bm25_hits[0: top_k]:
            print(
                f"{ hit['score'] : .3f} \t {texts[hit['corpus_id']].replace("\n", " ") }"   #replace是把換行符號 \n 換成空格
            )

#=======================ReRank==============================================================
        print(f"\n Top3 hits by rerank-API({len(bm25_hits)} BM25 hits re-ranked)")
        docs = [texts[hit['corpus_id']] for hit in bm25_hits]
        rerank_result = cohere_client.rerank(query = query, documents = docs, top_n = top_k, return_documents = True)   #逐步匹配query跟每一個初步篩選後的文本
        # print(rerank_result.results)
        for hit in rerank_result.results:
            # print(hit)
            print(f"{hit.relevance_score : .3f} \t {hit.document.text.replace("\n", " ")}")


def rerank_search(query, texts, cohere_client):
    print("\n Simply Only processed with Rerank Search:")
    rerank_results = cohere_client.rerank(query = query, documents = texts, top_n = 3, return_documents = True)
    for idx, results in enumerate(rerank_results.results):
        print(idx, results.relevance_score, results.document.text)
    

def grounded_generation(query, cohere_client, results):            
    docs_dict = [{'text' : text} for text in results['texts']]
    # print(docs_dict)
    response = cohere_client.chat(
        message=query,
        documents=docs_dict,
    )
    return response


def local_rag(texts):
    llm = LlamaCpp(
            model_path = "../Phi-3-mini-4k-instruct-q4.gguf",
            n_gpu_layers = -1,
            max_tokens = 500,
            n_ctx = 2048,
            seed = 42,
            verbose=False, 
        )
    
    embedding_mdoel = HuggingFaceEmbeddings(               #將文本轉換為數值表達法的嵌入模型
        model_name='thenlper/gte-small',
    )

    db = FAISS.from_texts(                         #建立本地端向量資料庫
        texts,
        embedding_mdoel,
    )

    return llm, embedding_mdoel, db


def prompt_template():
    #{context} 檢索到的資料 (固定寫法)
    #{question} 放使用者的問題 (固定寫法)
    template = """
        <|user|>
        Relevant information:
        {context}                             
    
        Provide a concise answer to the following question using the relevant information provided above:
        {question}
        <|end|>
        <|assistant|>
    """
    
    prompt = PromptTemplate(
        template = template,
        input_variables = ["context", "question"],
    )
    return prompt


def local_rag_generation(llm, db, prompt, query):

    rag = RetrievalQA.from_chain_type(                #RAG流程
        llm = llm,
        retriever = db.as_retriever(search_kwargs={"k": 6}),
        chain_type = 'stuff',                         #stuff 這條鏈預設把文件填進名為 context 的變數，把問題填進 question
        chain_type_kwargs = {
            "prompt" : prompt
        },
        verbose = True
    ) 

    response = rag.invoke(
        {"query" : query}
    )
    return response


def main():
    api_key = "Zx1TGFnBlYHN9I7MPo4WJGVQKFND68Arl6IwDuWl"
    model = "embed-v4.0"
    query = 'how precise was the science'
    query_2 = "How much income did Interstellar generate?"
   
    cohere_client = load_cohere(api_key)                         #建立一個可以跟 Cohere API 溝通的客戶端物件
    texts = load_texts()
    texts_embeds = texts_embedding(cohere_client, texts, model)  #對文本發送到Cohere API  進行embedding 
    index = build_search_index(texts_embeds)                     #在搜尋之前須先透過儲存向量資料庫FAISS 來儲存並優化embedding向量 建立搜尋索引
    results = search(cohere_client, model, query, index, texts)  #密集檢索 Dense Retrival
    # print(f"Query: '{query}'\n Nearest neighbors:")
    # print(results)

    '''
    #======================關鍵字檢索=======================================================
    tokenized_corpus = texts_tokenize(texts)                      #關鍵字檢索收先需要將texts token化
    bm25 = BM25Okapi(tokenized_corpus)                            #把所有 tokenized documents 交給 BM25，建立一個 BM25 搜尋器
     
    keyword_and_rerank_search(query, bm25, texts, cohere_client)  #關鍵字檢索 with 重新排名器
    #======================================================================================
    '''

    # rerank_search(query, texts, cohere_client)                    #Simply Only processed with Rerank Search:

    results = search(cohere_client, model, query_2, index, texts) 
    cohere_response = grounded_generation(query_2, cohere_client, results)   #基於事實的生成  透過搜尋系統找出相關文本的結果之後 將問題與文本送到cohere client.caht提供生成
    print(cohere_response.text)

    llm, embedding_mdoel, db = local_rag(texts)                         #使用本地端RAG, 載入生成模型,嵌入模型,向量資料庫
    prompt = prompt_template()
    rag_response = local_rag_generation(llm, db, prompt, query_2)
    print(rag_response['result'])


if __name__ == "__main__":
    main()
