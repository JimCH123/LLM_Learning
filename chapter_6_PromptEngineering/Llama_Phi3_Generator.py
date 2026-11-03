import torch
import gc
import json
#llama-cpp-python 是讓 Python 程式可以呼叫並使用高效能 C++ AI 推論引擎 llama.cpp 的專案套件
#支援 GGUF 格式：能夠載入經過量化（Quantization）壓縮的 GGUF 模型檔案
from llama_cpp.llama import Llama  


class Phi3GGUF_Generator:
    """用 llama-cpp-python 載入 Phi-3-mini 的 GGUF 量化模型，並產生 JSON 格式的回答：
    釋放 GPU 記憶體 -> 載入模型 -> 產生 JSON -> 整理成好讀的縮排 -> 關閉模型。
    """

    REPO_ID = "microsoft/Phi-3-mini-4k-instruct-gguf"
    FILENAME = "*q4.gguf"     #filename：要抓哪個 .gguf 模型檔

    def __init__(self):
        self.llm = None

    # ---------- 步驟 1：釋放前面用過的 GPU 記憶體 ----------
    def free_gpu_memory(self):
        # del model, tokenizer, pipe  #把 Python 變數名稱移除
        gc.collect()                  #要求 Python 做 garbage collection, 清理已經沒有被引用的 Python 物件
        torch.cuda.empty_cache()      #釋放記憶體 把沒在使用的 GPU cache 釋放掉

    # ---------- 步驟 2：載入 GGUF 模型 ----------
    def load_model(self):
        self.llm = Llama.from_pretrained(
            repo_id = self.REPO_ID,
            filename = self.FILENAME,
            n_gpu_layers = -1,
            n_ctx = 2048,
            verbose = False,
        )

    # ---------- 步驟 3：產生回答 ----------
    def generate(self, prompt):
    # prompt 由呼叫的人傳入；用 return 回傳模型的回答（一段 JSON 字串）。
    #llm.create_chat_completion(...) 回傳的不是單純一段文字，
    #而是一個巢狀 dictionary，裡面包含模型回答、使用量、結束原因等資訊。
        output = self.llm.create_chat_completion(
            messages = [
                {
                    "role" : "user",
                    "content" : prompt
                },
            ],
            response_format={"type" : "json_object"},
            temperature = 0,
        )['choices'][0]['message']['content']    
        return output

    # ---------- 步驟 4：把 JSON 字串整理成好讀的縮排 ----------
    # 這個函式不需要用到 self（資料都由參數傳入），所以用 @staticmethod
    @staticmethod
    def format_json(output):
        return json.dumps(json.loads(output), indent=4)

    # ---------- 步驟 5：對範例提示詞產生 JSON 並印出 ----------
    def generate_example(self, prompt):
        output = self.generate(prompt)
        print(output)
        json_output = self.format_json(output)
        print(json_output)

    # ---------- 步驟 6：關閉模型 ----------
    def close(self):
        self.llm.close()

    # ---------- 一次跑完整個流程 ----------
    def run(self):
        self.free_gpu_memory()
        self.load_model()


if __name__ == "__main__":
    generator = Phi3GGUF_Generator()
    generator.run()
    
    prompt = "Create a warrior for an RPG in JSON format with the keys: name, class, level, attributes, equipment, inventory."
    
    try:
        generator.generate_example(prompt)
    finally:
        generator.close()