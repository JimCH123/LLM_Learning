from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_community.chat_models import ChatLlamaCpp


class Phi3ChatAgent:
    """用 LangChain 的 create_agent + 本機 Phi-3-mini（GGUF）做有記憶的多輪對話：
    載入模型 -> 建立 agent（含 checkpointer）-> 提問 -> 關閉模型。
    每個步驟是一個方法，共用的物件存在 self 裡。"""

    MODEL_PATH = "../Phi-3-mini-4k-instruct-q4.gguf"

    # 對話編號：同一個 thread_id 的呼叫算同一場對話（共用歷史），換一個編號就是全新的對話。
    THREAD_ID = "1"

    # 範例問題（原本程式裡的兩個問題），只是示範用；
    # 實際使用時請用 ask(question) 傳入你要問的問題。
    EXAMPLE_QUESTIONS = (
        "Hello, My name is Jim. What is 1+1?",
        "What is my name?",
    )

    def __init__(self):
        self.llm = None
        self.agent = None
        self.thread_config = {"configurable": {"thread_id": self.THREAD_ID}}

    # ---------- 步驟 1：載入模型 ----------
    def load_model(self):
        self.llm = ChatLlamaCpp(
            model_path = self.MODEL_PATH,
            n_gpu_layers = -1,
            max_tokens = 500,
            n_ctx = 2048,
            seed = 42,
            verbose = False,
            stop = ["<|end|>"]
        )

    # ---------- 步驟 2：建立 agent（checkpointer 負責記住歷史） ----------
    def build_agent(self):
        self.agent = create_agent(
            model = self.llm,
            tools = [],
            checkpointer = InMemorySaver(),
        )

    # ---------- 步驟 3：提問 ----------
    # question 由呼叫的人傳入；用 return 回傳模型的回答（字串）。
    # 回答在最後一則訊息的 .content。
    def ask(self, question):
        result = self.agent.invoke(
            {"messages": [{"role": "user", "content": question}]},
            self.thread_config,
        )
        return result["messages"][-1].content

    # ---------- 範例：用內建的範例問題依序問並印出結果 ----------
    def ask_example(self):
        for question in self.EXAMPLE_QUESTIONS:
            print(self.ask(question))

    # ---------- 步驟 4：關閉模型 ----------
    # LangChain 的 ChatLlamaCpp 本身沒有 close()，底層的 llama-cpp-python 物件在 .client
    def close(self):
        self.llm.client.close()

    # ---------- 一次跑完整個流程 ----------
    def run(self):
        self.load_model()
        self.build_agent()
        try:
            self.ask_example()
        finally:
            self.close()


if __name__ == "__main__":
    demo = Phi3ChatAgent()
    demo.run()