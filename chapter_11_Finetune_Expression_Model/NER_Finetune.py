from datasets import load_dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer
import evaluate
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
from transformers import pipeline


class NERFinetuner:
    """用 CoNLL-2003 微調 BERT 做命名實體辨識：每個步驟是一個方法，共用的物件存在 self 裡。"""

    def __init__(self):
        # 在不同步驟之間要共用的東西，先在這裡宣告
        self.dataset = None
        self.label2id = None
        self.id2label = None
        self.model_id = None
        self.tokenizer = None
        self.model = None
        self.tokenized = None
        self.seqeval_metric = None
        self.data_collator = None
        self.training_args = None
        self.trainer = None

    # ---------- 步驟 1：載入資料集 ----------
    def load_data(self):
        self.dataset = load_dataset("eriktks/conll2003", trust_remote_code=True)

    # ---------- 步驟 2：建立標籤對照表 ----------
    def build_label_maps(self):
        self.label2id = {label: i for i, label in enumerate(self.dataset["train"].features["ner_tags"].feature.names)}
        self.id2label = {i: label for label, i in self.label2id.items()}

    # ---------- 步驟 3：載入預訓練模型與分詞器 ----------
    def load_model(self):
        self.model_id = "bert-base-cased"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_id,
            num_labels=len(self.id2label),
            id2label=self.id2label,
            label2id=self.label2id,
        )

    # ---------- 步驟 4：分詞並對齊標籤 ----------
    # 用 staticmethod：函式裡不需要 self，tokenizer 改由參數傳進來，
    # 這樣 datasets 的 map 在計算快取指紋時，不會把整個 self（含模型）一起序列化。
    @staticmethod
    def align_labels(example, tokenizer):
        token_ids = tokenizer(
            example["tokens"],
            truncation = True,
            is_split_into_words = True,
        )
        labels = example["ner_tags"]

        updated_labels = []
        for index, label in enumerate(labels):
            word_ids = token_ids.word_ids(batch_index = index)
            previous_word_idx = None
            label_ids = []

            for word_idx in word_ids:
                if word_idx != previous_word_idx:
                    previous_word_idx = word_idx
                    updated_label = -100 if word_idx is None else label[word_idx]
                    label_ids.append(updated_label)

                elif word_idx is None:
                    label_ids.append(-100)

                else:
                    updated_label = label[word_idx]
                    if updated_label % 2 == 1:
                        updated_label += 1
                    label_ids.append(updated_label)

            updated_labels.append(label_ids)
        token_ids["labels"] = updated_labels
        return token_ids

    def tokenize_data(self):
        self.tokenized = self.dataset.map(
            self.align_labels,
            batched = True,
            fn_kwargs = {"tokenizer": self.tokenizer},
        )

    # ---------- 步驟 5：載入評估指標 ----------
    def load_metric(self):
        self.seqeval_metric = evaluate.load("seqeval")

    # ---------- 步驟 6：定義度量函式（會傳給 Trainer） ----------
    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = predictions.argmax(axis=-1)

        true_predictions = [
            [self.id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.id2label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = self.seqeval_metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    # ---------- 步驟 7：建立資料整理器 ----------
    def build_collator(self):
        self.data_collator = DataCollatorForTokenClassification(self.tokenizer)

    # ---------- 步驟 8：設定訓練參數 ----------
    def build_args(self):
        self.training_args = TrainingArguments(
            output_dir = "NER_Finetune_model",
            learning_rate = 2e-5,
            per_device_train_batch_size = 16,
            per_device_eval_batch_size = 16,
            num_train_epochs = 1,
            weight_decay = 0.01,
            save_strategy = "epoch",
            report_to = "none",
        )

    # ---------- 步驟 9：建立 Trainer ----------
    def build_trainer(self):
        self.trainer = Trainer(
            model = self.model,
            args = self.training_args,
            train_dataset = self.tokenized["train"],
            eval_dataset = self.tokenized["test"],
            data_collator = self.data_collator,
            compute_metrics = self.compute_metrics,
        )

    # ---------- 步驟 10：訓練、評估、儲存 ----------
    def train(self):
        self.trainer.train()
        result = self.trainer.evaluate()
        print(result)
        self.tokenizer.save_pretrained("NER_Finetune_model")
        self.model.save_pretrained("NER_Finetune_model")

    # ---------- 步驟 11：用存好的模型做預測 ----------
    # 這個方法只依賴資料夾 "NER_Finetune_model"，不依賴前面的步驟，
    # 所以資料夾存在時，可以單獨呼叫，不必重新訓練。
    def predict(self, text = "My name is Maarten and I live in Amsterdam."):
        token_classifier = pipeline(
            "token-classification",
            model = "NER_Finetune_model",
            tokenizer = "NER_Finetune_model",
        )

        pred = token_classifier(text)
        print(pred)

    # ---------- 一次跑完整個流程 ----------
    def run(self):
        self.load_data()
        self.build_label_maps()
        self.load_model()
        self.tokenize_data()
        self.load_metric()
        self.build_collator()
        self.build_args()
        self.build_trainer()
        self.train()
        self.predict()


if __name__ == "__main__":
    ner_finetuner = NERFinetuner()
    ner_finetuner.run()
