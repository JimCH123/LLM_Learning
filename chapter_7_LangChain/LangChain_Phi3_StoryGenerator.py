from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough


class Phi3LangChainDemo:
    """用 LangChain 串接 llama.cpp 載入的 Phi-3-mini（GGUF），分步驟生成一個故事：
    標題 -> 主角敘述 -> 故事。
    每一段 chain 都會「保留前面所有欄位，再多加一個新欄位」，所以後面的提示詞
    可以直接用前面產生的 {title}、{character}。"""

    MODEL_PATH = "../Phi-3-mini-4k-instruct-q4.gguf"

    #包含多個提示詞的鍊 生成一個故事包括 標題 主角敘述 故事摘要
    # 注意：三引號字串裡的縮排、換行都會變成提示詞的一部分，所以模板內容靠左寫。
    TITLE_TEMPLATE = """
<s><|user|>
Create a title of a story about {summary}. Only return the title.
<|end|>
<|assistant|>
"""
    CHARACTER_TEMPLATE = """
<s><|user|>
Describe the main character of a story about {summary} with the title {title}.
Use only two sentences.
<|end|>
<|assistant|>
"""

    STORY_TEMPLATE= """
<s><|user|>
Create a story about {summary} with the title {title}. The main character is: {character}.
Only return the story and it cannot be longer than one paragraph.
<|end|>
<|assistant|>
"""

    def __init__(self):
        self.llm = None
        self.title_prompt = None
        self.character_prompt = None
        self.story_prompt = None
        self.title_chain = None
        self.character_chain = None
        self.story_chain = None
        self.llm_chain = None

    # ---------- 步驟 1：載入模型 ----------
    def load_model(self):
        self.llm = LlamaCpp(
            model_path = self.MODEL_PATH,
            n_gpu_layers = -1,
            max_tokens = 500,
            n_ctx = 2048,
            seed = 42,
            verbose = False,
            stop = ["<|end|>"],
            temperature = 0
        )

    # ---------- 步驟 2：建立提示詞模板 ----------
    def build_prompt(self):
        self.title_prompt = PromptTemplate.from_template(template=self.TITLE_TEMPLATE)
        self.character_prompt = PromptTemplate.from_template(template=self.CHARACTER_TEMPLATE)
        self.story_prompt = PromptTemplate.from_template(template=self.STORY_TEMPLATE)

    # ---------- 步驟 3：串成 chain ----------
    # RunnablePassthrough.assign(名字 = 小任務)：
    #   把輸入的 dict 原樣保留，再多加一個欄位「名字」，內容是這個小任務的輸出。
    #   欄位名字就相當於舊版 LLMChain 的 output_key。
    def build_chain(self):
        self.title_chain = RunnablePassthrough.assign(title = self.title_prompt | self.llm)
        self.character_chain = RunnablePassthrough.assign(character = self.character_prompt | self.llm)
        self.story_chain = RunnablePassthrough.assign(story = self.story_prompt | self.llm)

        #把這些小任務接成完整工作流程
        self.llm_chain = self.title_chain | self.character_chain | self.story_chain

    # ---------- 步驟 4：只跑到標題 ----------
    # 回傳 dict：{'summary': ..., 'title': ...}
    def ask_title(self, input_prompt):
        return self.title_chain.invoke({"summary" : input_prompt})

    # ---------- 步驟 5：跑到主角（會自動先生成標題） ----------
    # 回傳 dict：{'summary': ..., 'title': ..., 'character': ...}
    def ask_character(self, input_prompt):
        return (self.title_chain | self.character_chain).invoke({"summary" : input_prompt})

    # ---------- 步驟 6：跑完整流程（標題 -> 主角 -> 故事） ----------
    # 回傳 dict：{'summary', 'title', 'character', 'story'}
    def ask_story(self, input_prompt):
        return self.llm_chain.invoke({"summary" : input_prompt})

    # ---------- 一次跑完整個流程 ----------
    def run(self):
        self.load_model()
        self.build_prompt()
        self.build_chain()


if __name__ == "__main__":
    demo = Phi3LangChainDemo()
    demo.run()

    summary = "a girl that lost her mother"

    title_response = demo.ask_title(input_prompt=summary)
    print(title_response)

    character_response = demo.ask_character(input_prompt=summary)
    print(character_response)

    story_response = demo.ask_story(input_prompt=summary)
    print(story_response)