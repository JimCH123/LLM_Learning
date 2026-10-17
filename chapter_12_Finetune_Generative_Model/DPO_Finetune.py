from datasets import load_dataset
from peft import AutoPeftModelForCausalLM
from transformers import BitsAndBytesConfig, AutoTokenizer
import torch
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import DPOConfig, DPOTrainer
from peft import PeftModel
from transformers import pipeline


class DPOFinetuner:
    """以 SFT 模型為起點做 DPO 偏好對齊：每個步驟是一個方法，共用的物件存在 self 裡。"""

    def __init__(self):
        # 在不同步驟之間要共用的東西，先在這裡宣告
        self.dpo_dataset = None
        self.bnb_config = None
        self.model = None
        self.merged_model = None
        self.model_name = None
        self.tokenizer = None
        self.peft_config = None
        self.model_for_dpo = None
        self.training_arguments = None
        self.dpo_trainer = None
        self.sft_model = None
        self.dpo_model = None

    # ---------- 步驟 1：載入偏好資料集並過濾 ----------
    def load_data(self):
        self.dpo_dataset = load_dataset(
            "argilla/distilabel-intel-orca-dpo-pairs",
            split="train",
        )

        self.dpo_dataset = self.dpo_dataset.filter(
            lambda example:
            example["status"] != "tie" and
            example["chosen_score"] >= 8 and
            not example["in_gsm8k_train"]
        )

    # ---------- 步驟 2：整理成 prompt / chosen / rejected ----------
    # 用 staticmethod：函式裡不需要 self，這樣 datasets 的 map 在計算快取指紋時，
    # 不會把整個 self（含模型）一起序列化。
    @staticmethod
    def format_prompt(example):
        system = "<|system|> \n" + example["system"] + "</s> \n"
        prompt = "<|user|> \n" + example["input"] + "</s> \n <|assistant|> \n"
        chosen = example["chosen"] + "</s> \n"
        rejected = example["rejected"] + "</s> \n"
        return {
            "prompt": system + prompt,
            "chosen": chosen,
            "rejected": rejected,
        }

    def prepare_dataset(self):
        self.dpo_dataset = self.dpo_dataset.map(
            self.format_prompt,
            remove_columns=self.dpo_dataset.column_names,
        )

        # print(self.dpo_dataset[0])

    # ---------- 步驟 3：以 4 位元載入 SFT 模型並合併 ----------
    def load_sft_model(self):
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit = True,
            bnb_4bit_quant_type = "nf4",
            bnb_4bit_use_double_quant = True,
            bnb_4bit_compute_dtype = torch.bfloat16
        )

        self.model = AutoPeftModelForCausalLM.from_pretrained(
            "Qlora_Finetune_model",
            device_map="auto",
            low_cpu_mem_usage=True,
            quantization_config = self.bnb_config,
        )
        self.merged_model = self.model.merge_and_unload()

    # ---------- 步驟 4：載入分詞器 ----------
    def load_tokenizer(self):
        self.model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code = True,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

    # ---------- 步驟 5：設定新的 LoRA，並為量化訓練做準備 ----------
    def prepare_model(self):
        self.peft_config = LoraConfig(
            r=64,
            lora_alpha=32,
            target_modules=["k_proj", "gate_proj", "up_proj", "q_proj", "v_proj", "o_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model_for_dpo = prepare_model_for_kbit_training(self.merged_model)

    # ---------- 步驟 6：設定 DPO 訓練參數 ----------
    def build_training_args(self):
        self.training_arguments = DPOConfig(
            output_dir = "Dpo_results",
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            optim = "paged_adamw_32bit",
            learning_rate = 1e-5,
            lr_scheduler_type = "cosine",
            max_steps = 200,
            logging_steps = 10,
            bf16 = True,
            fp16 = False,
            gradient_checkpointing = True,
            max_length = 512,
            beta = 0.1,
        )

    # ---------- 步驟 7：建立 DPOTrainer ----------
    def build_trainer(self):
        self.dpo_trainer = DPOTrainer(
            self.model_for_dpo,
            args = self.training_arguments,
            train_dataset = self.dpo_dataset,
            processing_class = self.tokenizer,
            peft_config = self.peft_config,
        )

    # ---------- 步驟 8：訓練並儲存 DPO 附加層 ----------
    def train(self):
        self.dpo_trainer.train()
        self.dpo_trainer.save_model("Dpo_Finetune_model")

    # ---------- 步驟 9：合併 SFT 與 DPO，得到最終模型 ----------
    # 原本程式把變數 model 重新指定成新載入的模型；
    # 這裡改用區域變數 peft_model，避免蓋掉 self.model（訓練用的 4 位元模型）。
    # 注意：PeftModel.from_pretrained 與 merge_and_unload 會直接修改 sft_model 的權重，
    # 做完之後 self.sft_model 裡已經包含 DPO 的成果，不再是純 SFT 版本。
    def merge_models(self):
        peft_model = AutoPeftModelForCausalLM.from_pretrained(
            "Qlora_Finetune_model",
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        self.sft_model = peft_model.merge_and_unload()

        self.dpo_model = PeftModel.from_pretrained(
            self.sft_model,
            "Dpo_Finetune_model",
            device_map = "auto",
        )
        self.dpo_model = self.dpo_model.merge_and_unload()

    # ---------- 步驟 10：用最終模型生成回答 ----------
    # prompt 預設就是原本程式裡的那一句；不帶參數呼叫時行為完全一樣。
    def generate(self, prompt="""<|user|>
Tell me something about Large Language Models.</s>
<|assistant|>
"""):
        pipe = pipeline(task="text-generation", model=self.dpo_model, tokenizer=self.tokenizer)
        print(pipe(prompt)[0]["generated_text"])

    # ---------- 一次跑完整個流程 ----------
    def run(self):
        self.load_data()
        self.prepare_dataset()
        self.load_sft_model()
        self.load_tokenizer()
        self.prepare_model()
        self.build_training_args()
        self.build_trainer()
        self.train()
        self.merge_models()
        self.generate()


if __name__ == "__main__":
    finetuner = DPOFinetuner()
    finetuner.run()
