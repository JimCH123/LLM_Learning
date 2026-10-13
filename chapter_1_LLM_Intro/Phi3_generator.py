from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline


class Phi3Generator:
    """載入 Phi-3-mini 指令模型並生成回答。"""

    def __init__(self):
        # 在不同步驟之間要共用的東西，先在這裡宣告
        self.model = None
        self.tokenizer = None
        self.generator = None

    # ---------- 步驟 1：載入模型 ----------
    def load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3-mini-4k-instruct",
            device_map="cuda",
            torch_dtype="auto",
        )

    # ---------- 步驟 2：載入分詞器 ----------
    def load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/Phi-3-mini-4k-instruct",
        )

    # ---------- 步驟 3：建立生成管線 ----------
    def build_generator(self):
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            return_full_text=False,    #只回傳結果 不包含提示詞
            max_new_tokens=500,
            do_sample=False,           #模型是否使用採樣策略選擇下一個字元
        )

    # ---------- 一次跑完整個流程 ----------
    def run(self):
        self.load_model()
        self.load_tokenizer()
        self.build_generator()
        

    # ---------- 步驟 4：生成並印出回答 ----------
    # message 預設就是原本程式裡的那一則訊息；不帶參數呼叫時行為完全一樣。
    # 用 None 當預設值再在裡面建立，是因為 list 不適合直接當作函式的預設值
    # （預設值只會建立一次，會被所有呼叫共用）。
    def generate(self, message=None):
        if message is None:
            message = [
                {
                    "role": "user", 
                    "content": "Create a funny joke about chicken?"
                }
            ]

        output = self.generator(message)
        response = output[0]['generated_text']

        return response


if __name__ == "__main__":
    phi3_generator = Phi3Generator()
    phi3_generator.run()

    message = [
        {"role": "user", "content": "What is RAG?"}
    ]
    
    response = phi3_generator.generate(message)
    print(response)
