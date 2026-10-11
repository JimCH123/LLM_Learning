import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map = "cuda",
    torch_dtype = "auto",
)

tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct"
)

pipe = pipeline(
    "text-generation",
    model = model,
    tokenizer = tokenizer,
    return_full_text = False,
    max_new_tokens = 500,
    do_sample = False,
)

message = [
    {
        "role" : "user",
        "content" : "Create a funny joke about chickens."
    }
]

output = pipe(message)
print(output[0]["generated_text"])

'''
#===================================================================
#訊息轉換為特定的提示詞模板
prompt = pipe.tokenizer.apply_chat_template(message, tokenize=False) 
print(prompt)
#===================================================================

#===================================================================
#Temperatur提高選擇不太可能的字元的機率
output = pipe(message, do_sample=True, temperature=1)
print(output[0]["generated_text"])
#===================================================================

#===================================================================
#top_p(採樣)控制LLM可考慮的詞元子集大小
output = pipe(message, do_sample=True, top_p=1)
print(output[0]["generated_text"])
#===================================================================

#===================================================================
#情境學習： 提供範例
one_shot_prompt = [
    {
        "role" : "user",
        "content" : "A 'Gigamuru' is a type of Japanese musical instrument. "
        "An example of a sentence that uses the word Gigamuru is: "
    },
    {
        "role" : "assistant",
        "content" : "I have a Gigamuru that my uncle gave me as a gift. I love to play it at home. "
    },
    {
        "role" : "user",
        "content" : "To 'screeg' something is to swing a sword at it. An example of a sentence that uses the word screeg is:"
    },
]

print(tokenizer.apply_chat_template(one_shot_prompt, tokenize=False))
output = pipe(one_shot_prompt)
print(output[0]["generated_text"])
#===================================================================


#===================================================================
#連鎖提示： 將一個prompt的輸出作為下一個promt的輸入
product_prompt = [
    {
        "role" : "user",
        "content" : "Create a name and slogan for a chatbot that leverages LLM."
    },
]

output = pipe(product_prompt)
product_description = output[0]['generated_text']
print(product_description)

sales_prompt = [
    {
        "role" : "user",
        "content" : f"Generate a very shot sales pitch for the following product: {'product_description'}"
    },
]
output = pipe(sales_prompt)
sales_pitch = output[0]['generated_text']
print(sales_pitch)
#===================================================================
'''

import gc  
import torch
del model, tokenizer, pipe  #把 Python 變數名稱移除
 
# gc.collect()                #要求 Python 做 garbage collection, 清理已經沒有被引用的 Python 物件
# torch.cuda.empty_cache()    #釋放記憶體 把沒在使用的 GPU cache 釋放掉

from llama_cpp.llama import Llama  #llama-cpp-python 是讓 Python 程式可以呼叫並使用高效能 C++ AI 推論引擎 llama.cpp 的專案套件
                                   #支援 GGUF 格式：能夠載入經過量化（Quantization）壓縮的 GGUF 模型檔案
import json

llm = Llama.from_pretrained(
    repo_id = "microsoft/Phi-3-mini-4k-instruct-gguf",
    filename = "*fp16.gguf",     #filename：要抓哪個 .gguf 模型檔
    n_gpu_layers = -1,
    n_ctx = 2048,
    verbose = False,
)

output = llm.create_chat_completion(
    messages = [
        {"role" : "user",
         "content" : "Create a warrior for an RPG in JSON format."},
    ],
    response_format={"type" : "json_object"},
    temperature = 0,
)['choices'][0]['message']['content']    #llm.create_chat_completion(...) 回傳的不是單純一段文字，
                                          #而是一個巢狀 dictionary，裡面包含模型回答、使用量、結束原因等資訊。
print(output)
json_output = json.dumps(json.loads(output), indent=4)
print(json_output)

