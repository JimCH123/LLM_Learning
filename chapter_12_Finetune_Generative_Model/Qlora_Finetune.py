from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
import torch
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from peft import AutoPeftModelForCausalLM
from transformers import pipeline


class QLoRAFinetuner:
    """用 QLoRA 微調 TinyLlama 基礎模型成聊天助理：每個步驟是一個方法，共用的物件存在 self 裡。"""

    def __init__(self):
        # 在不同步驟之間要共用的東西，先在這裡宣告
        self.model_id = None
        self.template_tokenizer = None
        self.dataset = None
        self.model_name = None
        self.bnb_config = None
        self.model = None
        self.tokenizer = None
        self.peft_config = None
        self.training_args = None
        self.trainer = None
        self.merged_model = None

    # ---------- 步驟 1：載入「借來用」的對話模板分詞器 ----------
    def load_template_tokenizer(self):
        self.model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

        self.template_tokenizer = AutoTokenizer.from_pretrained(self.model_id)

    # ---------- 步驟 2：把一筆對話整理成 text ----------
    # 用 staticmethod：函式裡不需要 self，分詞器改由參數傳進來，
    # 這樣 datasets 的 map 在計算快取指紋時，不會把整個 self 一起序列化。
    @staticmethod
    def format_prompt(example, template_tokenizer):

        chat = example["messages"]
        prompt = template_tokenizer.apply_chat_template(chat, tokenize=False)
        return {"text": prompt}

    # ---------- 步驟 3：載入資料集並整理成只剩 text 欄位 ----------
    def prepare_dataset(self):
        self.dataset = (
            load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft")
                .shuffle(seed=42)
                .select(range(3000))
        )
        self.dataset = self.dataset.map(
            self.format_prompt,
            remove_columns=self.dataset.column_names,
            fn_kwargs={"template_tokenizer": self.template_tokenizer},
        )

    # ---------- 步驟 4：4 位元量化並載入基礎模型 ----------
    def load_model(self):
        self.model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"

        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=self.bnb_config,
            device_map="auto",
        )

        self.model.config.use_cache = False
        self.model.config.pretraining_tp = 1

    # ---------- 步驟 5：載入訓練用的分詞器 ----------
    def load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

    # ---------- 步驟 6：設定 LoRA ----------
    def build_lora_config(self):
        self.peft_config = LoraConfig(
            r=64,
            lora_alpha=32,
            target_modules=["k_proj", "gate_proj", "up_proj", "q_proj", "v_proj", "o_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

    # ---------- 步驟 7：設定訓練參數（SFTConfig） ----------
    def build_training_args(self):
        output_dir = "./Qlora_results"
        self.training_args = SFTConfig(
            output_dir=output_dir,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            optim="paged_adamw_32bit",
            learning_rate=2e-4,
            bf16=True,
            fp16=False,
            lr_scheduler_type="cosine",
            num_train_epochs=1,
            logging_steps=10,
            gradient_checkpointing=True,
            max_length=512,
            dataset_text_field="text",
        )

    # ---------- 步驟 8：建立 SFTTrainer ----------
    def build_trainer(self):
        self.trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.dataset,
            args=self.training_args,
            peft_config=self.peft_config,
            processing_class=self.tokenizer,
        )

    # ---------- 步驟 9：訓練並儲存 LoRA 附加層 ----------
    def train(self):
        self.trainer.train()
        self.trainer.save_model("Qlora_Fineune_model")

    # ---------- 步驟 10：載入 LoRA 並合併回基礎模型 ----------
    # 原本程式把變數 model 重新指定成新載入的模型；
    # 這裡改用區域變數 peft_model，避免蓋掉 self.model（訓練用的 4 位元模型）。
    def merge_lora(self):
        peft_model = AutoPeftModelForCausalLM.from_pretrained(
            "Qlora_Fineune_model",
            device_map="auto",
            low_cpu_mem_usage=True,
        )

        self.merged_model = peft_model.merge_and_unload()

    # ---------- 步驟 11：用合併後的模型生成回答 ----------
    # prompt 預設就是原本程式裡的那一句；不帶參數呼叫時行為完全一樣。
    def generate(self, prompt="""
<|user|>
Tell me something about Large Language Models (LLMs).
</s>
<|assistant|>
"""):
        pipe = pipeline(
            "text-generation",
            model=self.merged_model,
            tokenizer=self.tokenizer,
        )

        print(pipe(prompt)[0]["generated_text"])

    # ---------- 一次跑完整個流程 ----------
    def run(self):
        self.load_template_tokenizer()
        self.prepare_dataset()
        self.load_model()
        self.load_tokenizer()
        self.build_lora_config()
        self.build_training_args()
        self.build_trainer()
        self.train()
        self.merge_lora()
        self.generate()


if __name__ == "__main__":
    finetuner = QLoRAFinetuner()
    finetuner.run()
