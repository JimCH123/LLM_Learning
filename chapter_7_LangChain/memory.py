from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain
from langchain_classic.memory import ConversationBufferMemory, ConversationBufferWindowMemory, ConversationSummaryMemory


llm = LlamaCpp(
    model_path = "../Phi-3-mini-4k-instruct-fp16.gguf",
    n_gpu_layers = -1,
    max_tokens = 500,
    n_ctx = 2048,
    seed = 42,
    verbose = False,
    stop = ["<|end|>"]
)

template = """
<s><|user|>
Use the conversation summary below to answer the user's latest message.

Conversation summary:
{chat_history}

Latest user message:
{input_prompt}

Answer the latest user message based on the conversation summary when relevant.
<|end|>
<|assistant|>
"""

prompt = PromptTemplate(
    template= template,
    input_variables= ["chat_history", "input_prompt"]
)

'''
#=============================================================
#對話緩衝區 提醒他過去發生的事情
memory = ConversationBufferMemory(
    memory_key= "chat_history"
)

#將LLM 與提示詞和記憶串連在一起
llm_chain = LLMChain(
    llm = llm,
    prompt = prompt,
    memory = memory,
)

response = llm_chain.invoke(
    {
        "input_prompt" : "Hello, My name is Jim. What is 1+1?"
    }
)
print(response)

history_response = llm_chain.invoke(
    {
        "input_prompt" : "What is my name?"
    }
)
print(history_response)
'''

'''
#==================================================
#ConversationBufferyWindowMemory 視窗式對話緩衝區 只保存最近k次對話 減少上下文視窗大小的方法
memory = ConversationBufferWindowMemory(
    k = 2,                                  #只保存2次對話內容
    memory_key= "chat_history",
)

llm_chain = LLMChain(
    llm = llm,
    prompt = prompt,
    memory = memory,
)

#llm_chain.predict() 和 llm_chain.invoke() 的核心區別在於：
# invoke 是 LangChain 現代表達式語言（LCEL）的標準通用介面，而 predict 則是舊版的語法
# invoke輸入格式為dict, predict輸入格式為kwargs (variable="value") 且輸出直接為string

response_1 = llm_chain.predict(
    input_prompt = "Hello, My name is Jim and I am 24 years old. What is 1+1?"
)
response_2 = llm_chain.predict(
    input_prompt = "What is 3+3?"
)

print(response_2)

history_response_1 = llm_chain.predict(
    input_prompt = "What is my name?"
)
print(history_response_1)   #到這邊LLM記住了response_1, response_2 所以還可以回答我的名字

history_response_2 = llm_chain.predict(
    input_prompt = "What is my age?"
)
print(history_response_2)  #到這邊LLM只記住了response_2, history_response_1 已經忘記我的年紀

'''

'''
#============================================================
#summary_prompt_template 定義摘要規則 要怎麼把舊摘要 + 新對話整理成新的摘要
#summary = 過去濃縮後的記憶
#new_lines = 最新發生的對話
#兩者合起來，才能產生更新後的記憶。

summary_prompt_template = """                                       
<s><|user|>
Update the conversation summary using the new conversation lines.

Preserve important information such as:
- the user's name and personal details they explicitly provide
- questions the user asked
- important facts and preferences
- previous answers when relevant

Existing summary:
{summary}

New conversation:
{new_lines}

Return only the updated summary.
<|end|>
<|assistant|>
"""

#用 PromptTemplate 把摘要文字模板變成 LangChain 可使用的 prompt
summary_prompt = PromptTemplate( 
    template= summary_prompt_template,
    input_variables= ["new_lines", "summary"],
)

#ConversationSummaryMemory 需要令一個LLM來執行總結
#把這個 summary_prompt 和一個 llm 交給 ConversationSummaryMemory  持續更新 conversation summary
memory = ConversationSummaryMemory(
    llm = llm,
    memory_key= "chat_history",   #指定這份 memory 對外回傳時使用的 dictionary key 提供給主對話 Prompt 使用。
    prompt= summary_prompt,
)

llm_chain = LLMChain(
    llm = llm,
    prompt = prompt,
    memory = memory,
)

response_1 = llm_chain.invoke(
    {
        "input_prompt" : "Hi. My name is Jim. What is 1+1?"
    }
)

response_2 = llm_chain.invoke(
    {
        "input_prompt" : "What is my name?"
    }
)
print(response_2)

response_3 = llm_chain.invoke(
    {
        "input_prompt" : "What was the first question I asked??"
    }
)
print(response_3)
print(memory.load_memory_variables({}))  #檢查目前的摘要內容
#=======================================================================================
'''
