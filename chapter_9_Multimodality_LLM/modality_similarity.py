#計算單一圖片和單一caption的相似度
#CLIP model : openai/clip-vit-base-patch32
#venv312

from urllib.request import urlopen
from PIL import Image
from transformers import CLIPTokenizer, CLIPProcessor, CLIPModel
import torch
import numpy as np
import matplotlib.pyplot as plt

def cos_similarity(text_embedding, image_embedding):
    text_embedding = text_embedding.pooler_output / text_embedding.pooler_output.norm(dim = -1, keepdim = True)
    image_embedding = image_embedding.pooler_output / image_embedding.pooler_output.norm(dim = -1, keepdim = True)

    text_embedding = text_embedding.detach().cpu().numpy()   #PyTorch Tensor 轉成 NumPy .detach()分離梯度
    image_embedding = image_embedding.detach().cpu().numpy() 

    score = text_embedding @ image_embedding.T            #@矩陣乘法
    return score


def show_preprocessed_image(processed_image):
    img = processed_image.squeeze(0)                #用來移除陣列或張量（Tensor）中第 0 維度且大小為 1 的維度
    #將張量（Tensor，例如圖片）的維度順序完全顛倒（反轉）即 (C, H, W) 變為 (W, H, C)
    #torch.arange(img.ndim - 1, -1, -1) 從 img.ndim - 1 開始、到 -1 結束、步長為 -1 的反向索引序列 : [2, 1, 0]
    # *(星號運算子) Python 的解包（Unpacking）運算子，把 torch.arange 產生的張量或串列打開，變成個別的引數傳入 permute()
    #img.permute(...) 重新排列維度 把原本第 2 維變成第 0 維、第 1 維保持不變、原本第 0 維變成第 2 維
    img = img.permute(*torch.arange(img.ndim - 1, -1, -1)) 
    img = np.einsum('ijk->jik', img)

    # Visualize preprocessed image
    plt.imshow(img)
    plt.axis('off')
    plt.show()


def preprocess_embedding(caption, image, clip_tokenizer, clip_processer, model):
    inputs = clip_tokenizer(           #caption先做分詞
        caption,
        return_tensors = "pt",
    )
    # print(inputs)
    # tokens = clip_tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])   #查看token_id 轉換為詞元
    # print(tokens)
    text_embedding = model.get_text_features(**inputs)    #取得caption的embedding


    processed_image = clip_processer(                   #影像預處理
        image,
        text = None,
        return_tensors = "pt",
    )['pixel_values']
    print(processed_image.shape) 

    # show_preprocessed_image(processed_image)  

    image_embedding = model.get_image_features(         #取得image的embedding
    processed_image,
    )
    print(image_embedding.pooler_output.shape) 
    return text_embedding, image_embedding


def load_model(model_id):
    model = CLIPModel.from_pretrained(model_id)
    clip_tokenizer = CLIPTokenizer.from_pretrained(model_id)
    clip_processer = CLIPProcessor.from_pretrained(model_id)
    return model, clip_tokenizer ,clip_processer


def main():
    puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
    image = Image.open(urlopen(puppy_path)).convert("RGB")
    # image.show()
    caption = "a puppy playing in the snow"
    model_id = "openai/clip-vit-base-patch32"

    model, clip_tokenizer ,clip_processer = load_model(model_id)
    text_embedding, image_embedding = preprocess_embedding(caption, image, clip_tokenizer, clip_processer, model)
    score = cos_similarity(text_embedding, image_embedding)
    print(score)


if __name__ == "__main__":
    main()