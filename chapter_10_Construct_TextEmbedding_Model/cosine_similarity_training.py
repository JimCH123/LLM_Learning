# Embedding model: SentenceTransformer("bert-base-uncased")
# 僅僅將 softmax loss 改成用 Cosine Similarity 作為loss function 建構 句子嵌入模型

from datasets import load_dataset, Dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer import losses
from sentence_transformers.sentence_transformer.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.sentence_transformer.training_args import SentenceTransformerTrainingArguments
from sentence_transformers.sentence_transformer.trainer import SentenceTransformerTrainer
import mteb


def LoadDataset():
    train_dataset = load_dataset(
        path = "nyu-mll/glue",
        name = "mnli",
        split = "train",
    ).select(range(50000))
    # print(train_dataset[0])
    train_dataset = train_dataset.remove_columns("idx")
    mapping = {0:1, 1:0, 2:0}
    train_dataset = Dataset.from_dict(
        sentences1 = train_dataset['premise'],
        sentences2 = train_dataset['hypothesis'],
        labels = [mapping[label] for label in train_dataset['label']]
    )
    return train_dataset


def Define_train_model():
    #載入預訓練transformer 用於embedding文字的基礎模型
    embedding_model = SentenceTransformer("bert-base-uncased")

    #定義loss function 可使用softmax
    cossim_train_loss = losses.CosineSimilarityLoss(
        model = embedding_model,
    ) 

    #定義訓練參數
    #output_dir : 訓練輸出資料夾，存放 `checkpoint-xxx/`、訓練記錄
    #num_train_epochs : 完整跑過訓練資料幾輪 建議增加輪數 
    #per_device_train_batch_size : 訓練時每張 GPU 每一步處理 32 筆資料
    #per_device_eval_batch_size : 評估時每張 GPU 每次處理 32 筆資料
    #warmup_steps : 前 100 步學習率從 0 線性升到設定值，避免一開始更新太劇烈而破壞預訓練權重
    #fp16 : 使用 16 位元混合精度訓練，速度更快、更省記憶體
    #eval_steps :  每 100 步做一次評估 
    #logging_steps : 每 100 步記錄一次 loss 等訓練資訊
    args = SentenceTransformerTrainingArguments(
        output_dir = "base_embedding_model",
        num_train_epochs = 1,          
        per_device_train_batch_size = 32,   
        per_device_eval_batch_size = 32,
        warmup_steps = 100,
        fp16 = True,
        eval_steps = 100,
        logging_steps = 100,
    )
    return embedding_model, cossim_train_loss, args

    

def Define_evaluator():
    #建立embedding相似性 評估器 通常使用 語意文本相似性基準(STSB）
    #載入評估所使用的dataset
    eval_dataset = load_dataset(
        path = "nyu-mll/glue",
        name = "stsb",
        split="validation",
    )
    # print(eval_dataset[0])

    evaluator = EmbeddingSimilarityEvaluator(
        sentences1 = eval_dataset['sentence1'],
        sentences2 = eval_dataset['sentence2'],
        scores = [score / 5 for score in eval_dataset['label']],
        main_similarity = "cosine",
    )
    return evaluator


def Load_trainer(embedding_model, args, train_dataset, train_loss, evaluator):

    trainer = SentenceTransformerTrainer(
        model = embedding_model,
        args = args,
        train_dataset = train_dataset,
        loss = train_loss,
        evaluator = evaluator  
    )
    print("before:", evaluator(embedding_model))

    return trainer


def mteb_evaluation(embedding_model):
    mteb_task = mteb.get_tasks(tasks=["Banking77Classification"])
    mteb_results = mteb.evaluate(embedding_model, tasks = mteb_task)
    for result in mteb_results.task_results:
        print(f"\nTask: {result.task_name}")
        print(f"Main score: {result.main_score}")
        for split, subset_results in result.scores.items():
            print(f"\n{split}:")
            for subset_result in subset_results:
                for metric, score in subset_result.items():
                    print(f"  {metric}: {score}")


def main():
    #生成對比範例 首先載入dataset: GLUE裡的NLI
    train_dataset = LoadDataset()    
    print(train_dataset[0])

    #定義訓練模型 所需的 基礎embedding model, loss function, train arguments
    embedding_model, cossim_train_loss, args = Define_train_model()


    #定義評估器 可使用GLUE裡的STSB 來評估模型在語意相似性任務上的評分
    stsb_evaluator = Define_evaluator()

    #載入訓練模型
    trainer = Load_trainer(embedding_model, args, train_dataset, cossim_train_loss, stsb_evaluator)
    #開始訓練
    trainer.train()

    #評估訓練結果
    print(stsb_evaluator(embedding_model))

    mteb_evaluation(embedding_model)



if __name__ == "__main__":
    main()

