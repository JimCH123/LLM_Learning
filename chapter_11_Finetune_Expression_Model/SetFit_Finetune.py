from setfit import sample_dataset, SetFitModel, TrainingArguments, Trainer
from datasets import load_dataset


class SetFitFinetuner:
    """SetFit 少樣本微調流程。"""

    def __init__(self):
        self.tomatoes = None
        self.sampled_train_data = None
        self.test_data = None
        self.model = None
        self.args = None
        self.trainer = None

    # ---------- 步驟 1：載入資料並取樣 ----------
    def load_data(self):
        self.tomatoes = load_dataset("cornell-movie-review-data/rotten_tomatoes")
        self.sampled_train_data = sample_dataset(self.tomatoes["train"], num_samples = 16)
        self.test_data = self.tomatoes["test"]

    # ---------- 步驟 2：載入 SetFit 模型 ----------
    def load_model(self):
        self.model = SetFitModel.from_pretrained(
            "sentence-transformers/all-mpnet-base-v2"
        )

    # ---------- 步驟 3：設定訓練參數 ----------
    def build_args(self):
        self.args = TrainingArguments(
            output_dir = "SetFit_Finetune_model",
            num_epochs = 3,
            num_iterations = 20,
        )
        # self.args.eval_strategy = self.args.evaluation_strategy

    # ---------- 步驟 4：建立 Trainer ----------
    def build_trainer(self):
        self.trainer = Trainer(
            model = self.model,
            args = self.args,
            train_dataset = self.sampled_train_data,
            eval_dataset = self.test_data,
            metric = "f1",
        )

    # ---------- 步驟 5：訓練、儲存、評估 ----------
    def train(self):
        self.trainer.train()
        self.model.save_pretrained(self.args.output_dir)
        result = self.trainer.evaluate()
        print(f"After training : {result} ")

    # ---------- 一次跑完整個流程 ----------
    def run(self):
        self.load_data()
        self.load_model()
        self.build_args()
        self.build_trainer()
        self.train()


if __name__ == "__main__":
    finetuner = SetFitFinetuner()
    finetuner.run()
