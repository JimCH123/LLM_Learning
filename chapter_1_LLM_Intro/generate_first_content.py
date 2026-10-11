from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map="cuda",
    torch_dtype="auto",                
    )

tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,    #只回傳結果 不包含提示詞
    max_new_tokens=500,
    do_sample=False,           #模型是否使用採樣策略選擇下一個字元
)

message = [
    {"role": "user", "content": "Create a funny joke about chicken?"}
]

output = generator(message)
print(output[0]['generated_text'])