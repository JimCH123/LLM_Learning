#Run in .venv312
import os

from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI


os.environ["OPENAI_API_KEY"] = "API_KEY"

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
)

'''
#=======================================================================================
#template的原本寫法  手動告訴 LangChain 模板需要這四個輸入變數。
react_template = """
Answering the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you shuld always think about what to do
Action: the action shoud take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
...(This Though/ Action/ Action Input/ Observation can be repeat N times) 
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}
"""

prompt = PromptTemplate(
    template= react_template,
    input_variables= ["tools", "tool_names", "input", "agent_scratchpad"]
)
#=======================================================================================
'''

#from_template()  LangChain 會自動從 {...} 裡辨識需要的變數
#{tools}{tool_names}{input}{agent_scratchpad}LangChain 的 ReAct Agent 跟 Prompt 之間約定好的 interface 不能自己亂取名
# {tools}:create_react_agent 自動把工具描述放進去
# {tool_names}:create_react_agent 自動放工具名稱
# {agent_scratchpad}:AgentExecutor 自動放前面 Action / Observation 告訴模型目前已有的推理/工具執行紀錄，然後繼續完成任務

prompt = PromptTemplate.from_template("""
Answer the following question as best you can.

You have access to these tools:

{tools}                                                     

Use this format:

Question: the input question
Thought: think about what to do
Action: one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
...(This Though/ Action/ Action Input/ Observation can be repeat N times) 
Thought: I now know the final answer
Final Answer: the final answer

Question: {input}
Thought: {agent_scratchpad}
""")


search = DuckDuckGoSearchRun()      #從 DuckDuckGoSearchRun 這個 class 建立一個 search 物件（instance）

#把 search.run 這個普通 Python 方法，「包裝」成 LangChain Agent 可以理解和使用的 Tool
search_tool = Tool(
    name="duckduckgo_search",       #Agent 之後就是使用這個 name
    description="Search the web for current information.",   #Agent 看 description 判斷什麼時候應該用這個工具
    func=search.run,                 #search 物件的實體方法
)

#llm-math 本身是一個 LangChain tool
#如果這個工具本身需要使用 LLM 就傳給我建立的llm
tools = load_tools(["llm-math"], llm=llm)  
tools.append(search_tool)

#建構react代理 讓 Agent「知道有哪些工具可以選」
react_agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
)

#讓 Executor「真的能執行那些工具」
agent_executor = AgentExecutor(
    agent=react_agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)

response = agent_executor.invoke(
    {
        "input": "What is the current price of a Iphone 18 pro is USD? How much would it cost in TWD if the exchange rate is 31.78 NTD for 1 USD."
    }
)

print(response["output"])


#output: 
# > Entering new AgentExecutor chain...
# Thought: I need to find out the current price of an iPhone 18 pro in USD and then calculate the cost in TWD using the exchange rate.
# Action: duckduckgo_search
# Action Input: "current price of iPhone 18 pro in USD"Sep 9, 2026 · Get $35–$885 off a new iPhone 18 Pro or iPhone 18 Pro Max when you trade in an iPhone 8 Plus or newer. 0% financing available. Buy or lease now with free shipping. 1 day ago · How Much Does the iPhone 18 Pro Cost? Apple sells the iPhone 18 Pro in four storage capacities. The base 256GB model is $1,199 (about 5 full-time workdays at $30 per hour), followed by 512GB at $1,399, 1TB at $1,799, and 2TB at $2,399, a ladder independently matched in a current retail price guide. The jump between tiers is not uniform. Apple iPhone 18 Pro is a 5G iOS 27 Smartphone powered by Apple A20 Pro with 12 GB RAM and 256 GB storage. Price starts at $1,439 in United States. iPhone 18 Pro Pro further. From $1199 or $49.95/mo. for 24 mo. * Lease from $34.99/mo. for 24 mo. with Apple Upgrade Buy 4 days ago · Compare iPhone 18 Pro pricing worldwide and find the cheapest country for iPhone 18 Pro. See how iPhone 18 Pro costs vary by region, including Turkey, United States, and more countries. 6 days ago · Apple announced the iPhone 18 Pro and Pro Max starting at $1,199 and $1,299, with the 2nm A20 Pro, a variable aperture camera and 45-hour battery. Sep 10, 2026 · Order Apple iPhone 18 Pro today at Verizon. Learn about the new features, reviews, colors, prices, and more. iPhone 18 Pro Pro further. From $1199 or $49.95/mo. for 24 mo. * Lease from $34.99/mo. for 24 mo. with Apple Upgrade Buy 4 days ago · Compare iPhone 18 Pro pricing worldwide and find the cheapest country for iPhone 18 Pro. See how iPhone 18 Pro costs vary by region, including Turkey, United States, and more countries.Action: Calculator
# Action Input: 1199 * 31.78Answer: 38104.22Final Answer: The current price of an iPhone 18 Pro in USD is $1,199. It would cost 38,104.22 TWD based on the exchange rate of 31.78 NTD for 1 USD.

# > Finished chain.
# The current price of an iPhone 18 Pro in USD is $1,199. It would cost 38,104.22 TWD based on the exchange rate of 31.78 NTD for 1 USD.