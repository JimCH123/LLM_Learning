import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

class Phi3PromptingDemo:
    """用 Phi-3-mini 示範幾種常見的提示技巧：
    基本生成 -> 聊天模板 -> temperature -> top_p -> 情境學習(one-shot) -> 連鎖提示。
    """

    MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"

    def __init__(self):
        self.model = None
        self.tokenizer = None

        self.message = [
            {
                "role" : "user",
                "content" : "Create a funny joke about chickens."
            }
        ]
            
    # ---------- 步驟 1：載入模型 ----------    
    def load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_NAME,
            device_map = "cuda",
            torch_dtype = "auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_NAME,
        )

    # ---------- 步驟 2：建立生成管線 ----------
    def build_pipe(self):
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            return_full_text = False,
            max_new_tokens = 500,
            do_sample = False,
        )

    # ---------- 步驟 3：基本生成 ---------------
    def basic_generation(self):
        output = self.pipe(self.message)
        print(output[0]["generated_text"])

    # ---------- 步驟 4：看聊天模板長什麼樣子 ----------
    def show_chat_template(self):
        #訊息轉換為特定的提示詞模板
        prompt = self.pipe.tokenizer.apply_chat_template(self.message, tokenize=False)
        print(prompt)
        
    # ---------- 步驟 5：temperature ----------
    def temperature_demo(self):
        #Temperatur提高選擇不太可能的字元的機率
        output = self.pipe(self.message, do_sample=True, temperature=1)
        print(output[0]["generated_text"])
 
    # ---------- 步驟 6：top_p ----------
    def top_p_demo(self):
        #top_p(採樣)控制LLM可考慮的詞元子集大小
        output = self.pipe(self.message, do_sample=True, top_p=1)
        print(output[0]["generated_text"])

    # ---------- 步驟 7：情境學習 提供範例（one-shot） ----------
    def in_context_learning(self):
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
 
        print(self.tokenizer.apply_chat_template(one_shot_prompt, tokenize=False))
        output = self.pipe(one_shot_prompt)
        print(output[0]["generated_text"])


    # ---------- 步驟 8：連鎖提示 將一個prompt的輸出作為下一個promt的輸入----------
    def chain_prompting(self):
        product_prompt = [
            {
                "role" : "user",
                "content" : "Create a name and slogan for a chatbot that leverages LLM."
            },
        ]
 
        output = self.pipe(product_prompt)
        product_description = output[0]['generated_text']
        print(product_description)
 
        sales_prompt = [
            {
                "role" : "user",
                "content" : f"Generate a very short sales pitch for the following product: {product_description}"
            },
        ]
        output = self.pipe(sales_prompt)
        sales_pitch = output[0]['generated_text']
        print(sales_pitch)


    def run(self):
        self.load_model()
        self.build_pipe()
        # self.basic_generation()
        # self.show_chat_template()
        # self.temperature_demo()
        # self.top_p_demo()
        # self.in_context_learning()
        self.chain_prompting()



if __name__ == "__main__":
    demo = Phi3PromptingDemo()
    demo.run()