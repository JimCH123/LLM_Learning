#採用 all-MiniLM-L6-v2 已經是訓練好的句向量模型，
# 相較於 bert-base-uncased 只是通用的語言模型底座。
from datasets import load_dataset
from sentence_transformers.sentence_transformer.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.sentence_transformer import losses
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.training_args import SentenceTransformerTrainingArguments
from sentence_transformers.sentence_transformer.trainer import SentenceTransformerTrainer

def LoadDataset():
    train_dataset = load_dataset(
        "nyu-mll/glue",
        "mnli",
        split = "train",
    ).select(range(50000))
    train_dataset = train_dataset.remove_columns("idx")
    # print(train_dataset[0])
    return train_dataset


def load_evaluator():
    val_sts = load_dataset(
        "nyu-mll/glue",
        "stsb",
        split = "validation",
    )
    # print(val_sts)
    stsb_evaluator = EmbeddingSimilarityEvaluator(
        sentences1 = val_sts["sentence1"],
        sentences2 = val_sts["sentence2"],
        scores = [score / 5 for score in val_sts["label"]],
        main_similarity = "cosine",
    )
    return stsb_evaluator


def define_train_model():
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    mnr_loss = losses.MultipleNegativesRankingLoss(
        model = embedding_model,
    )
    args = SentenceTransformerTrainingArguments(
        output_dir = "mnr_embedding_model",
        num_train_epochs = 1,
        per_device_train_batch_size = 32,
        per_device_eval_batch_size = 32,
        warmup_steps = 100,
        fp16 = True,
        eval_steps = 100,
        logging_steps = 100,  
    )
    return embedding_model, mnr_loss, args


def load_trainer(embedding_model, mnr_loss, args, train_dataset, stsb_evaluator):
    trainer = SentenceTransformerTrainer(
        model = embedding_model,
        args = args,
        train_dataset = train_dataset,
        loss = mnr_loss,
        evaluator = stsb_evaluator,
    )
    return trainer


def main():
    train_dataset = LoadDataset()
    stsb_evaluator = load_evaluator()
    embedding_model, mnr_loss, args = define_train_model()
    trainer = load_trainer(embedding_model, mnr_loss, args, train_dataset, stsb_evaluator)
    trainer.train()
    print(stsb_evaluator(embedding_model))


if __name__ == "__main__":
    main()