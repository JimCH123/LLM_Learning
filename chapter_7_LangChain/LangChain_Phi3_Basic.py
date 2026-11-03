from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate


class Phi3LangChainDemo:
    """用 LangChain 串接 llama.cpp 載入的 Phi-3-mini（GGUF）：
    載入模型 -> 建立提示詞模板 -> 用 prompt | llm 串成 chain -> 傳入問題。
    """
    MODEL_PATH = "../Phi-3-mini-4k-instruct-q4.gguf"

    # 提示詞模板：{input_prompt} 是變數，呼叫 chain 時會被換成真正的問題。
    # Phi-3 的聊天格式是 <|user|> ... <|end|> <|assistant|>
    TEMPLATE = """
        <s><|user|>
        {input_prompt}
        <|end|>
        <|assistant|>
        """

    def __init__(self):
        self.llm = None
        self.prompt = None
        self.basic_chain = None

    # ---------- 步驟 1：載入模型 ----------
    def load_model(self):
        self.llm = LlamaCpp(
            model_path = self.MODEL_PATH,
            n_gpu_layers = -1,
            max_tokens = 500,
            n_ctx = 2048,
            seed = 42,
            verbose = False,
            stop = ["<|end|>"]
        )

    # ---------- 步驟 2：建立提示詞模板 ----------
    def build_prompt(self):
        self.prompt = PromptTemplate.from_template(template=self.TEMPLATE)

    # ---------- 步驟 3：串成 chain ----------
    def build_chain(self):
        self.basic_chain = self.prompt | self.llm

    # ---------- 步驟 4：把問題放進 chain 生成回答 ----------
    # input_prompt 由呼叫的人傳入。
    def ask(self, input_prompt):
        response = self.basic_chain.invoke(
            {
                "input_prompt" : input_prompt
            }
        )
        return response

    # ---------- 一次跑完整個流程 ----------
    def run(self):
        self.load_model()
        self.build_prompt()
        self.build_chain()


if __name__ == "__main__":
    demo = Phi3LangChainDemo()
    demo.run()

    input_prompt = "Hi, My name is Jim. What is 1 + 1?"
    response = demo.ask(input_prompt=input_prompt)
    print(response)