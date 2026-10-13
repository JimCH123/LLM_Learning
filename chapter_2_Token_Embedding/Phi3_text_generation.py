from transformers import AutoModelForCausalLM, AutoTokenizer

class Phi3Generator:
    def __init__(self, model_name = "microsoft/Phi-3-mini-4k-instruct", device = "cuda", max_new_tokens = 200):
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens

        self.model = None
        self.tokenizer = None

    def load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map = self.device,
            torch_dtype = "auto",
        )

    def load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name) 

    def run(self):
        self.load_model()
        self.load_tokenizer()

    # ---------- 生成的三個步驟 ----------
    def encode(self, prompt):
        #文字 → token id（張量)，並放到和模型相同的裝置上。
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        return input_ids
 
    def generate_ids(self, input_ids, max_new_tokens=None):
        #token id → 模型生成的 token id（包含輸入的提示詞部分）。
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens
 
        generation_output = self.model.generate(
            input_ids = input_ids,
            max_new_tokens=max_new_tokens
        )
        return generation_output
 
    def decode(self, generation_output):
        #token id → 文字。
        return self.tokenizer.decode(generation_output[0])
 
    # ---------- 對外的主要方法 ----------
    def generate(self, prompt=None, max_new_tokens=None):
        """把三個步驟串起來，回傳生成的文字。"""
        
        input_ids = self.encode(prompt)
        generation_output = self.generate_ids(input_ids, max_new_tokens)
        return self.decode(generation_output)



if __name__ == "__main__":
    phi3_generator = Phi3Generator()
    phi3_generator.run()

    prompt = """
    Write an email apologizing to Jim for the trajic gardening mishap. 
    Explain how it happened. <|assistant|>
    """

    response = phi3_generator.generate(prompt, max_new_tokens = 20)
    print(response)
