#計算單一圖片和單一caption的相似度
#CLIP model : openai/clip-vit-base-patch32
#venv312

from urllib.request import urlopen
from PIL import Image
from transformers import CLIPTokenizer, CLIPProcessor, CLIPModel
import torch
import numpy as np
import matplotlib.pyplot as plt


class ClipImageTextSimilarity:
    """用 CLIP 計算「一張圖片」和「一句 caption」的相似度（餘弦相似度）。

    CLIP 同時有「文字編碼器」和「圖片編碼器」，會把文字和圖片都變成同一個空間裡的向量（embedding），
    意思越接近，兩個向量的方向就越接近，餘弦相似度就越高。

    流程：載入圖片 -> 載入模型 -> 文字編碼 + 圖片編碼 -> 計算餘弦相似度。
    """

    MODEL_ID = "openai/clip-vit-base-patch32"
    IMAGE_URL = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
    CAPTION = "a puppy playing in the snow"

    def __init__(self):
        self.model = None
        self.clip_tokenizer = None
        self.clip_processer = None

    # ---------- 步驟 1：載入圖片 ----------
    def load_image(self, image_url):
        """從網址下載圖片，並轉成 RGB（三個顏色通道）。"""
        image = Image.open(urlopen(image_url)).convert("RGB")
        # image.show()
        return image

    # ---------- 步驟 2：載入模型 ----------
    def load_model(self):
        """載入三樣東西（都用同一個 MODEL_ID）：
        - model：CLIP 模型本體（裡面有文字編碼器與圖片編碼器）
        - clip_tokenizer：把文字切成 token、轉成 token id
        - clip_processer：圖片前處理（縮放、裁切、正規化等）
        """
        self.model = CLIPModel.from_pretrained(self.MODEL_ID)
        self.clip_tokenizer = CLIPTokenizer.from_pretrained(self.MODEL_ID)
        self.clip_processer = CLIPProcessor.from_pretrained(self.MODEL_ID)

    # ---------- 步驟 3：文字編碼 ----------
    def encode_text(self, caption):
        """caption（文字）-> 分詞 -> 文字的 embedding。"""
        inputs = self.clip_tokenizer(           #caption先做分詞
            caption,
            return_tensors = "pt",
        )
        # print(inputs)
        # tokens = self.clip_tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])   #查看token_id 轉換為詞元
        # print(tokens)
        text_embedding = self.model.get_text_features(**inputs)    #取得caption的embedding
        return text_embedding

    # ---------- 步驟 4：圖片編碼 ----------
    def encode_image(self, image):
        """image（圖片）-> 前處理 -> 圖片的 embedding。"""
        processed_image = self.clip_processer(                   #影像預處理
            image,
            text = None,
            return_tensors = "pt",
        )['pixel_values']
        print(processed_image.shape)

        # self.show_preprocessed_image(processed_image)

        image_embedding = self.model.get_image_features(         #取得image的embedding
            processed_image,
        )
        print(image_embedding.pooler_output.shape)
        return image_embedding

    # ---------- 步驟 5：計算相似度 ----------
    @staticmethod
    def cos_similarity(text_embedding, image_embedding):
        """計算餘弦相似度。
        先把向量除以自己的長度（變成長度 1 的單位向量），再做矩陣乘法（內積），結果就是餘弦相似度。
        """
        text_embedding = text_embedding.pooler_output / text_embedding.pooler_output.norm(dim = -1, keepdim = True)
        image_embedding = image_embedding.pooler_output / image_embedding.pooler_output.norm(dim = -1, keepdim = True)

        text_embedding = text_embedding.detach().cpu().numpy()   #PyTorch Tensor 轉成 NumPy .detach()分離梯度
        image_embedding = image_embedding.detach().cpu().numpy()

        score = text_embedding @ image_embedding.T            #@矩陣乘法
        return score

    def compute_similarity(self, caption, image):
        """把步驟 3、4、5 串起來：caption + image -> 相似度。
        caption 和 image 由呼叫的人傳入，所以可以換不同的文字與圖片重複計算。
        """
        text_embedding = self.encode_text(caption)
        image_embedding = self.encode_image(image)
        score = self.cos_similarity(text_embedding, image_embedding)
        return score

    # ---------- 輔助：看看前處理後的圖片（除錯／教學用，預設不會被呼叫）----------
    @staticmethod
    def show_preprocessed_image(processed_image):
        """把前處理後的圖片畫出來，看看模型真正「看到」的是什麼。"""
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

    # ---------- 一次跑完整個流程 ----------
    def run(self):
        image = self.load_image(self.IMAGE_URL)
        self.load_model()
        score = self.compute_similarity(self.CAPTION, image)
        print(score)


if __name__ == "__main__":
    demo = ClipImageTextSimilarity()
    demo.run()