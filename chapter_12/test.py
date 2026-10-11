from datasets import load_dataset
from peft import AutoPeftModelForCausalLM
from transformers import BitsAndBytesConfig, AutoTokenizer
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import DPOConfig, DPOTrainer
from peft import PeftModel
from transformers import pipeline


dpo_dataset = load_dataset(
    "argilla/distilabel-intel-orca-dpo-pairs",
    split="train",  
)

dpo_dataset = dpo_dataset.filter(
    lambda example:
        example["status"] != "tie" and
        example["chosen_score"] >= 8 and
        not example["in_gsm8k_train"]
)

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


dpo_dataset = dpo_dataset.map(
    format_prompt,
    remove_columns=dpo_dataset.column_names,
)

# print(dpo_dataset[0])

bnb_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_quant_type = "nf4",
    bnb_4bit_use_double_quant = True,
    bnb_4bit_compute_dtype = torch.bfloat16
)

model = AutoPeftModelForCausalLM.from_pretrained(
    "Qlora_Finetune_model",
    device_map="auto",
    low_cpu_mem_usage=True,
    quantization_config = bnb_config,
)
merged_model = model.merge_and_unload()

model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code = True,
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

peft_config = LoraConfig(
    r=64,
    lora_alpha=32,
    target_modules=["k_proj", "gate_proj", "up_proj", "q_proj", "v_proj", "o_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model_for_dpo = prepare_model_for_kbit_training(merged_model)


training_arguments = DPOConfig(
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

dpo_trainer = DPOTrainer(
    model_for_dpo,
    args = training_arguments,
    train_dataset = dpo_dataset,
    processing_class = tokenizer,
    peft_config = peft_config,
)
dpo_trainer.train()
dpo_trainer.save_model("Dpo_Finetune_model")

model = AutoPeftModelForCausalLM.from_pretrained(
    "Qlora_Finetune_model",
    device_map="auto",
    low_cpu_mem_usage=True,
)
sft_model = model.merge_and_unload()

dpo_model = PeftModel.from_pretrained(
    sft_model,
    "Dpo_Finetune_model",
    device_map = "auto",
)
dpo_model = dpo_model.merge_and_unload()


prompt = """<|user|>
Tell me something about Large Language Models.</s>
<|assistant|>
"""

pipe = pipeline(task="text-generation", model=dpo_model, tokenizer=tokenizer)
print(pipe(prompt)[0]["generated_text"])
