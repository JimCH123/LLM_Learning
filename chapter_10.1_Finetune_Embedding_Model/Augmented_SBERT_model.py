import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset, Dataset
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.sentence_transformer import losses
from sentence_transformers.sentence_transformer.datasets import NoDuplicatesDataLoader
from sentence_transformers.sentence_transformer.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.sentence_transformer.training_args import SentenceTransformerTrainingArguments
from sentence_transformers.sentence_transformer.trainer import SentenceTransformerTrainer


class AugmentedSBERTPipeline:
    """Augmented SBERT：gold 訓練老師 → 老師幫 silver 貼標籤 → 合併訓練學生。"""

    def __init__(self):
        # 在不同步驟之間要共用的東西，先在這裡宣告
        self.dataset = None          # gold 的原始 HF Dataset
        self.mapping = None          # 標籤對照表
        self.gold_dataloader = None  # 給 Cross-Encoder（老師）用
        self.gold = None             # 給最後合併用（DataFrame）
        self.cross_encoder = None
        self.silver_raw = None       # silver 的原始 HF Dataset
        self.pairs = None
        self.silver = None           # 老師貼完標籤的 DataFrame
        self.train_dataset = None
        self.evaluator = None
        self.embedding_model = None
        self.train_loss = None
        self.args = None
        self.trainer = None

    # ---------- 步驟 1：載入 gold 資料與標籤對照 ----------
    def load_gold(self):
        self.dataset = load_dataset(
            "nyu-mll/glue",
            "mnli",
            split ="train",
        ).select(range(10000))
        print(self.dataset[0])

        self.mapping = {0:1, 1:0, 2:0}

    # ---------- 步驟 2：gold 準備成兩種容器 ----------
    def build_gold(self):
        gold_examples = [
            InputExample(
                texts = [row["premise"], row["hypothesis"]],
                label = self.mapping[row["label"]],
            ) for row in tqdm(self.dataset)
        ]
        print(gold_examples[0])

        self.gold_dataloader = NoDuplicatesDataLoader(gold_examples, batch_size = 32)

        self.gold = pd.DataFrame(
            {
                "sentence1" : self.dataset["premise"],
                "sentence2" : self.dataset["hypothesis"],
                "label" : [self.mapping[label] for label in self.dataset["label"]],
            }
        )
        print(self.gold.head())

    # ---------- 步驟 3：訓練 Cross-Encoder（老師） ----------
    def train_cross_encoder(self):
        self.cross_encoder = CrossEncoder(
            "bert-base-uncased",
            num_labels = 2,
        )
        self.cross_encoder.fit(
            train_dataloader = self.gold_dataloader,
            epochs = 1,
            show_progress_bar = True,
            warmup_steps = 100,
            use_amp = False,
        )

    # ---------- 步驟 4：載入 silver 資料 ----------
    def load_silver(self):
        self.silver_raw = load_dataset(
            "nyu-mll/glue",
            "mnli",
            split ="train",
        ).select(range(10000, 50000))
        self.pairs = list(zip(self.silver_raw["premise"], self.silver_raw["hypothesis"]))
        print(self.pairs[0])

    # ---------- 步驟 5：老師幫 silver 貼標籤 ----------
    def label_silver(self):
        output = self.cross_encoder.predict(
            self.pairs,
            apply_softmax = True,
        )
        print(output)
        self.silver = pd.DataFrame(
            {
                "sentence1" : self.silver_raw["premise"],
                "sentence2" : self.silver_raw["hypothesis"],
                "label" : np.argmax(output, axis = 1)
            }
        )

    # ---------- 步驟 6：合併 gold + silver ----------
    def build_train_dataset(self):
        data = pd.concat(
            [self.gold, self.silver],
            axis = 0,
            ignore_index = True,
        )
        data = data.drop_duplicates(
            subset=["sentence1", "sentence2"],
            keep = "first",
        )
        self.train_dataset = Dataset.from_pandas(
            data,
            preserve_index = False
        )
        print(self.train_dataset[0])

    # ---------- 步驟 7：建立 STS-B 評估器 ----------
    def build_evaluator(self):
        val_sts = load_dataset(
            "nyu-mll/glue",
            "stsb",
            split = "validation",
        )
        self.evaluator = EmbeddingSimilarityEvaluator(
            sentences1=val_sts["sentence1"],
            sentences2=val_sts["sentence2"],
            scores = [score / 5 for score in val_sts["label"]],
            main_similarity="cosine",
        )

    # ---------- 步驟 8：建立 Bi-Encoder（學生）與 loss ----------
    def build_model(self):
        self.embedding_model = SentenceTransformer("bert-base-uncased")

        self.train_loss = losses.CosineSimilarityLoss(
            model = self.embedding_model,
        )

    # ---------- 步驟 9：建立訓練參數與 Trainer ----------
    def build_trainer(self):
        self.args = SentenceTransformerTrainingArguments(
            output_dir = "Augmented_SBERT_model",
            num_train_epochs = 1,
            per_device_train_batch_size = 32,
            per_device_eval_batch_size = 32,
            warmup_steps = 100,
            fp16 = True,
            eval_steps = 100,
            logging_steps = 100,
        )

        self.trainer = SentenceTransformerTrainer(
            model = self.embedding_model,
            args = self.args,
            train_dataset = self.train_dataset,
            loss = self.train_loss,
            evaluator = self.evaluator,
        )

    # ---------- 步驟 10：訓練、評估、儲存 ----------
    def train(self):
        print("before:", self.evaluator(self.embedding_model))
        self.trainer.train()
        print("after:", self.evaluator(self.embedding_model))

        self.embedding_model.save("Augmented_SBERT_model")

    # ---------- 一次跑完整個流程 ----------
    def run(self):
        self.load_gold()
        self.build_gold()
        self.train_cross_encoder()
        self.load_silver()
        self.label_silver()
        self.build_train_dataset()
        self.build_evaluator()
        self.build_model()
        self.build_trainer()
        self.train()


if __name__ == "__main__":
    pipeline = AugmentedSBERTPipeline()
    pipeline.run()
