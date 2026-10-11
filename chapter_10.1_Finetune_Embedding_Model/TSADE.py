import nltk
from tqdm import tqdm
from datasets import Dataset, load_dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer import losses
from sentence_transformers.sentence_transformer.datasets import DenoisingAutoEncoderDataset
from sentence_transformers.sentence_transformer.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.sentence_transformer.modules import Transformer, Pooling
from sentence_transformers.sentence_transformer.training_args import SentenceTransformerTrainingArguments
from sentence_transformers.sentence_transformer.trainer import SentenceTransformerTrainer


class TSDAEPipeline:
    """TSDAE 訓練流程：每個步驟是一個方法，共用的物件存在 self 裡。"""

    def __init__(self):
        # 在不同步驟之間要共用的東西，先在這裡宣告
        self.mnli = None
        self.flat_sentence = None
        self.train_dataset = None
        self.evaluator = None
        self.embedding_model = None
        self.train_loss = None
        self.args = None
        self.trainer = None

    # ---------- 步驟 1：載入資料並攤平 ----------
    def load_sentences(self):
        nltk.download('punkt_tab')

        self.mnli = load_dataset(
            "nyu-mll/glue",
            "mnli",
            split = "train",
        ).select(range(25000))

        self.flat_sentence = list(self.mnli["premise"]) + list(self.mnli["hypothesis"])
        print(self.flat_sentence[0])

    # ---------- 步驟 2：製造（壞句, 原句）資料集 ----------
    def build_train_dataset(self):
        damaged_data = DenoisingAutoEncoderDataset(
            list(set(self.flat_sentence))
        )

        train_dataset = {
            "damaged_data" : [],
            "original_data" : [],
        }

        for data in tqdm(damaged_data):
            train_dataset["damaged_data"].append(data.texts[0])
            train_dataset["original_data"].append(data.texts[1])

        self.train_dataset = Dataset.from_dict(
            train_dataset
        )
        print(self.train_dataset[0])

    # ---------- 步驟 3：建立 STS-B 評估器 ----------
    def build_evaluator(self):
        val_sts = load_dataset(
            "nyu-mll/glue",
            "stsb",
            split = "validation",
        )
        self.evaluator = EmbeddingSimilarityEvaluator(
            sentences1 = val_sts["sentence1"],
            sentences2 = val_sts["sentence2"],
            scores = [score / 5 for score in val_sts["label"]],
            main_similarity = "cosine"
        )

    # ---------- 步驟 4：建立 Bi-Encoder（CLS pooling） ----------
    def build_model(self):
        word_embedding_model = Transformer("bert-base-uncased")
        pooling_model = Pooling(
            embedding_dimension = word_embedding_model.get_embedding_dimension(),
            pooling_mode = "cls",
        )
        self.embedding_model = SentenceTransformer(modules = [word_embedding_model, pooling_model])
        self.embedding_model.max_seq_length = 128

    # ---------- 步驟 5：建立 DAE loss（含 decoder） ----------
    def build_loss(self):
        self.train_loss = losses.DenoisingAutoEncoderLoss(
            model = self.embedding_model,
            tie_encoder_decoder = True,
        )
        self.train_loss.decoder = self.train_loss.decoder.to("cuda")

    # ---------- 步驟 6：建立訓練參數與 Trainer ----------
    def build_trainer(self):
        self.args = SentenceTransformerTrainingArguments(
            output_dir = "TSADE_embedding_model",
            num_train_epochs = 1,
            per_device_train_batch_size = 16,
            per_device_eval_batch_size = 16,
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

    # ---------- 步驟 7：訓練、評估、儲存 ----------
    def train(self):
        print("before:", self.evaluator(self.embedding_model))
        self.trainer.train()
        print("after:", self.evaluator(self.embedding_model))
        self.embedding_model.save("TSADE_embedding_model")

    # ---------- 一次跑完整個流程 ----------
    def run(self):
        self.load_sentences()
        self.build_train_dataset()
        self.build_evaluator()
        self.build_model()
        self.build_loss()
        self.build_trainer()
        self.train()


if __name__ == "__main__":
    pipeline = TSDAEPipeline()
    pipeline.run()
