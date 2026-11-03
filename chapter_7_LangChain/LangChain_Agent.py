from langchain.agents import create_agent
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


class ToolAgent:
    """用 create_agent 讓 LLM 自己決定要用哪個工具（網路搜尋 / 計算機）來回答問題：
    載入金鑰與模型 -> 建立工具 -> 建立 agent -> 提問。
    """

    MODEL = "gpt-3.5-turbo"

    # 系統提示詞：新版不再需要 {tools}、{tool_names}、{agent_scratchpad} 這些佔位符，
    # 也不用自己寫 Thought / Action / Observation 的格式，工具資訊與呼叫流程都由 create_agent 處理。
    SYSTEM_PROMPT = "Answer the following question as best you can. "
    "Call tools one at a time. Wait for the search result before using the calculator, "
    "and never put placeholders in a calculator expression."

    def __init__(self):
        # 在不同步驟之間要共用的東西，先在這裡宣告
        self.llm = None
        self.search = None
        self.tools = None
        self.agent = None

    # ---------- 步驟 1：載入金鑰並建立模型 ----------
    # 金鑰放在 .env（OPENAI_API_KEY=...），load_dotenv() 讀進環境變數後，ChatOpenAI 會自己使用。
    def load_llm(self):
        load_dotenv()

        self.llm = ChatOpenAI(
            model=self.MODEL,
            temperature=0,
        )

    # ---------- 步驟 2：建立工具 ----------
    def build_tools(self):
        #從 DuckDuckGoSearchRun 這個 class 建立一個 search 物件（instance）
        #llm-math 本身是一個 LangChain tool
        #如果這個工具本身需要使用 LLM 就傳給我建立的llm
        self.search = DuckDuckGoSearchRun()      
        self.tools = load_tools(["llm-math"], llm=self.llm)
        self.tools.append(self.search)

    # ---------- 步驟 3：建立 agent ----------
    # 舊版要分兩步：create_react_agent（讓 Agent「知道有哪些工具可以選」）
    # 再包進 AgentExecutor（讓 Executor「真的能執行那些工具」）。
    # 新版的 create_agent 一次做完這兩件事。
    def build_agent(self):
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=self.SYSTEM_PROMPT,
        )

    # ---------- 步驟 4：提問 ----------
    def ask(self, question):
    # question 由呼叫的人傳入；用 return 回傳 agent 的最終答案（字串）。
    # 回傳的是 dict，裡面有整串訊息（問題、工具呼叫、工具結果、最終答案），
    # 最終答案在最後一則訊息的 .content。
        result = self.agent.invoke(
            {"messages": [{"role": "user", "content": question}]}
        )
        return result["messages"][-1].content

    # ---------- 一次跑完整個流程 ----------
    def run(self):
        self.load_llm()
        self.build_tools()
        self.build_agent()


if __name__ == "__main__":
    agent = ToolAgent()
    agent.run()

    question = "What is the current price of a Iphone 18 pro is USD? How much would it cost in TWD if the exchange rate is 31.78 NTD for 1 USD."
    response = agent.ask(question=question)
    print(response)