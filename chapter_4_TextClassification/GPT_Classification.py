import openai
from dotenv import load_dotenv
from openai import OpenAI


class ChatGPTSentimentClassifier:
    """用 OpenAI 的聊天模型（雲端 API）做影評情緒分類。"""

    # 提示詞模板：[DOCUMENT] 是佔位符，呼叫時會被換成真正的影評。
    # 常數放在類別層級，不要散落在程式各處。
    #[DOCUMENT] 只是我自己取的標記，模型看不到它，因為送出去之前它已經被換成真正的影評了。
    PROMPT = """Predict whether the following document is a positive or negative movie review.:

    [DOCUMENT]

    If it is a positive review, please respond only with "positive".
    If it is a negative review, please respond only with "negative".
    Please do not give any other answer.
    """

    # 範例影評（原本程式裡的 document），只是示範用的預設值；
    # 實際使用時請用 classify(document) 傳入你要分類的影評。
    DOCUMENT = "This movie is a great movie. The story is very interesting and the acting is superb. I would highly recommend it to anyone who enjoys a good film."

    def __init__(self):
        self.client = None

    # ---------- 步驟 1：載入金鑰並建立 client ----------
    def load_client(self):
        load_dotenv()
        self.client = OpenAI()

    # ---------- 步驟 2：呼叫模型 ----------
    def chatgpt_generation(self, prompt, document, model="gpt-3.5-turbo-0125"):
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a sentiment classifier. Respond only with 'positive' or 'negative'."},   #設定模型的整體角色/規則/行為
                {"role": "user", "content": prompt.replace("[DOCUMENT]", document)},
            ],
            temperature=0,
        )
        return response.choices[0].message.content    #response通常不是字串 而是很多欄位的回傳物件 可以透過print()查看

    # ---------- 步驟 3：對任意影評分類 ----------
    # document 由呼叫的人傳入；用 return 回傳結果，方便拿去比對、統計或存檔。
    def classify(self, document):
        return self.chatgpt_generation(self.PROMPT, document)

    # ---------- 範例：用內建的範例影評跑一次並印出結果 ----------
    def classify_example(self):
        chatgpt_response = self.classify(self.DOCUMENT)

        print(f"Example Return :{chatgpt_response}")

    # ---------- 一次跑完整個流程 ----------
    def run(self):
        self.load_client()
        self.classify_example()


if __name__ == "__main__":
    classifier = ChatGPTSentimentClassifier()
    classifier.run()
    response = classifier.classify("Such a terrible movie like a shit.")
    print(response)
