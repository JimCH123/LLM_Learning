import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

class Phi3Generator:
    def __init__(self, model_id = "microsoft/Phi-3-mini-4k-instruct", device = "cuda"):
        self.model_id = model_id
        self.device = device
        self.tokenizer = None
        self.model = None
        self.generator = None
        
    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map = self.device,
            torch_dtype="auto",
        )

    def build_generator(self):
        self.generator = pipeline(
            "text-generation",
            model = self.model,
            tokenizer = self.tokenizer,
            do_sample = False,
            return_full_text = False,  #只回傳結果 不包含提示詞
            max_new_tokens=50,
        )

    def generation1(self, prompt):
        output = self.generator(prompt)
        response = output[0]["generated_text"]
        return response

    def run(self):
        self.load_model()
        self.build_generator()



    def encode(self, prompt):
        input_ids = self.tokenizer(prompt, return_tensors = "pt")["input_ids"].to(self.device)
        return input_ids

    def generate_ids(self, input_ids):
        # print(self.model.named_parameters)
        model_output = self.model.model(input_ids = input_ids)
        # print(model_output)
        lm_head_output = self.model.lm_head(model_output[0])
        # print(lm_head_output)
        token_id = lm_head_output[0, -1].argmax(dim = -1)
        return token_id

    def decode(self, token_id):
        return self.tokenizer.decode(token_id)

    def generation2(self, prompt):
        input_ids = self.encode(prompt)
        token_id = self.generate_ids(input_ids)
        return self.decode(token_id)



if __name__ == "__main__":
    phi3_generator = Phi3Generator()
    phi3_generator.run()
    prompt = "The capital of France is"
    #=====================Method 1==========================
    response1 = phi3_generator.generation1(prompt)
    print(response1)
    #=======================================================

    #=====================Method 2==========================
    response2 = phi3_generator.generation2(prompt)
    print(response2)
    #=======================================================

