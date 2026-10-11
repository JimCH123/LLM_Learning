from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
from transformers import pipeline


class MLM_Finetune():
    def __init__(self):
        self.tomatoes = None
        self.train_data = None
        self.test_data = None
        self.model_id = None
        self.model = None
        self.tokenizer = None
        self.tokenized_train_data = None
        self.tokenized_test_data = None
        self.data_collator = None
        self.args = None
        self.trainer = None

    def load_data(self):
        self.tomatoes = load_dataset("cornell-movie-review-data/rotten_tomatoes")
        self.train_data, self.test_data = self.tomatoes["train"], self.tomatoes["test"]


    def load_model(self):
        self.model_id = "bert-base-uncased"
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)


    def tokenize_data(self):
        tokenzier = self.tokenizer
        self.tokenized_train_data = self.train_data.map(
            lambda x: tokenzier(x["text"], truncation=True), 
            batched=True
        ).remove_columns(["label"])

        self.tokenized_test_data = self.test_data.map(
            lambda x: tokenzier(x["text"], truncation=True), 
            batched=True
        ).remove_columns(["label"])


    def load_datacollator(self):
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=True,
            mlm_probability=0.15
        )


    def load_training_args(self):
        self.args = TrainingArguments(
            output_dir="MLM_Finetune_model",
            num_train_epochs=10,  
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            weight_decay=0.01,
            save_strategy="epoch",
            report_to="none",
        )   


    def train(self):
        self.trainer = Trainer(
            model=self.model,
            args=self.args,
            train_dataset=self.tokenized_train_data,
            eval_dataset=self.tokenized_test_data,
            data_collator=self.data_collator,
        )
        self.tokenizer.save_pretrained(self.args.output_dir)
        self.trainer.train()
        self.model.save_pretrained(self.args.output_dir)


    def prediction(self, model):
        mask_filler = pipeline(
            "fill-mask",
            model=model,
        )
        preds = mask_filler("What a horrible [MASK].")

        for pred in preds:
            print(f"{pred["sequence"]}, Score : {pred["score"] : .4f}")


    def run(self):
        self.load_data()
        self.load_model()
        self.tokenize_data()
        self.load_datacollator()
        self.load_training_args()
        self.train()
        print("Before Training : \n")
        self.prediction(model = self.model_id)
        print("\n After Training : \n")
        self.prediction(model = "MLM_Finetune_model")


if __name__ == "__main__" :
    mlm_finetuner = MLM_Finetune()
    mlm_finetuner.run()
