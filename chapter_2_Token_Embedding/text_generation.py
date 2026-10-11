from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map="cuda",
    torch_dtype="auto",
)

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct") 

prompt = "Write an email apologizing to Jim for the trajic gardening mishap. Explain how it happened. <|assistant|>" 

input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

generation_output = model.generate(
    input_ids = input_ids,
    max_new_tokens=20
)


print(tokenizer.decode(generation_output[0]))





