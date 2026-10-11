import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

#載入模型與分詞器
model_name = "microsoft/Phi-3-mini-4k-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cuda",  
    torch_dtype="auto",
)

#建立生成管道
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,  #只回傳結果 不包含提示詞
    max_new_tokens=50,
    do_sample=False,  #模型是否使用採樣策略選擇下一個字元
)

#定義提示詞
prompt = "The capital of France is"

#將提示詞轉換為模型輸入格式(分詞)
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

#lm前的輸出
model_output = model.model(input_ids=input_ids)

lm_model_output = model.lm_head(model_output[0])

token_id = lm_model_output[0,-1].argmax(dim=-1)  #取得最大機率的token id
print(tokenizer.decode(token_id))  #將token id轉換為文字

