from langchain_community.llms import LlamaCpp
from pathlib import Path
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain

PROJECT_DIR = Path(__file__).resolve().parent.parent

llm = LlamaCpp(
    model_path = "../Phi-3-mini-4k-instruct-fp16.gguf",
    n_gpu_layers = 20,
    max_tokens = 500,
    n_ctx = 2048,
    seed = 42,
    verbose = False,
    stop = ["<|end|>"]
)

'''
#=================================================================================================
#Langchain Practice 串接提示詞模板
response = llm.invoke("Hi, My name is Jim. What is 1 + 1?")   #Langchain 中使用invoke來生成輸出
print(response)               #不會顯示任何東西 因為Phi-3需要提示詞模板 因此透過langchain串接

template = """
<s><|user|>
{input_prompt}
<|end|>
<|assistant|>
"""

prompt = PromptTemplate(
    template = template,
    input_variables = ['input_prompt']
)

#===============================================================
# #PromptTemplate的第二種寫法 from_template會自動從字串模板找 {...} 變數
# prompt = PromptTemplate.from_template(template)
#===============================================================

basic_chain = prompt | llm       # "|" LangChain Expression Language，也就是 LCEL，用來把多個 runnable 串成一條 pipeline

response = basic_chain.invoke(
    {
        "input_prompt" : "Hi, My name is Jim. What is 1 + 1?"
    }
)
print(response)

#============================================================
'''

#===============================================================================
#包含多個提示詞的鍊 生成一個故事包括 標題 主角敘述 故事摘要
title_template = """
<s><|user|>
Create a title of a story about {summary}. Only return the title.
<|end|>
<|assistant|>
"""

title_prompt = PromptTemplate(
    template=title_template,
    input_variables=["summary"]
)

#LLMChain 是把 Prompt 和 LLM 包成一個可重複執行的小任務，並可以把輸出交給下一個 Chain 使用。
title = LLMChain(llm = llm, prompt = title_prompt, output_key = "title") 

# title_response = title.invoke(
#     {
#         "summary" : "a girl that lost her mother."
#     }
# )
# print(title_response)

character_template = """
<s><|user|>
Describe the main character of a story about {summary} with the title {title}.
Use only two sentences.
<|end|>
<|assistant|>
"""
character_prompt = PromptTemplate(
    template=character_template,
    input_variables=["summary", "title"],
)
character = LLMChain(llm = llm, prompt = character_prompt, output_key = "character")
# llm_chain = title | character
# character_response = llm_chain.invoke(
#     {
#         "summary":"a girl that lost her mother."
#     }
# )
# print(character_response)

story_template = """
<s><|user|>
Create a story about {summary} with the title {title}. The main character is: {character}.
Only return the story and it cannot be longer than one paragraph.
<|end|>
<|assistant|>
"""
story_prompt = PromptTemplate.from_template(template=story_template)
story = LLMChain(llm = llm, prompt = story_prompt, output_key = "story")


llm_chain = title | character | story                   #把這些小任務接成完整工作流程
story_response = llm_chain.invoke(
    {
        "summary":"a girl that lost her mother."
    }
)
print(story_response)

