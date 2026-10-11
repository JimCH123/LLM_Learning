from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
import numpy as np
import evaluate
from transformers import TrainingArguments, Trainer


class FinetuneBERT:
    def __init__(self):
        self.tomatoes = None
        self.train_data = None
        self.test_data = None
        self.model_id = None
        self.model = None
        self.tokenizer = None
        self.data_collator = None
        self.tokenized_train = None
        self.tokenized_test = None
        self.args = None
        self.trainer = None


    def load_data(self):
        self.tomatoes = load_dataset("cornell-movie-review-data/rotten_tomatoes")
        self.train_data, self.test_data = self.tomatoes["train"], self.tomatoes["test"]


    def load_model(self):
        # First Step : 載入基礎預訓練模型
        self.model_id = "google-bert/bert-base-cased"
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_id,
            num_labels = 2,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id
        )


    def tokenize_data(self):
        #對資料進行分詞處理
        #將多個獨立的資料樣本 組裝成一個 Batch（批次），以供機器學習模型
        tokenizer = self.tokenizer
        self.data_collator = DataCollatorWithPadding(tokenizer= tokenizer)

        self.tokenized_train = self.train_data.map(lambda x : tokenizer(x["text"]), batched = True)
        self.tokenized_test  =self.test_data.map(lambda x : tokenizer(x["text"]), batched = True)


    #定義度量指標
    @staticmethod
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis = -1)

        load_f1 = evaluate.load("f1")
        # print(load_f1)
        f1_scores = load_f1.compute(predictions=predictions, references=labels)["f1"]
        return {"f1" : f1_scores}

    def build_args(self):
        self.args = TrainingArguments(
            output_dir = "BERT_Finetune_model",
            learning_rate = 2e-5,
            per_device_train_batch_size = 16,
            per_device_eval_batch_size = 16,
            num_train_epochs = 1,
            weight_decay = 0.01,
            save_strategy = "epoch",
            report_to = "none"
        )


    def build_trainer(self):
        self.trainer = Trainer(
            model = self.model,
            args = self.args,
            data_collator = self.data_collator,
            train_dataset = self.tokenized_train,
            eval_dataset = self.tokenized_test,
            compute_metrics = self.compute_metrics,
        )


    def train(self):
        print("Before training:", self.trainer.evaluate())
        self.trainer.train()  
        self.trainer.save_model() 
        result = self.trainer.evaluate()
        print(f"After training : {result} ")


    def run(self):
        self.load_data()  
        self.load_model()  
        self.tokenize_data()
        self.build_args()
        self.build_trainer()
        self.train()
      


if __name__ == "__main__":
    finetuner = FinetuneBERT()
    finetuner.run()

