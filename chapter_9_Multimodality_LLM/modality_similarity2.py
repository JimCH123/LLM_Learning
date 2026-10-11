#CLIP model : clip-ViT-B-32 使用sentence transformer 載入clip 使的embedding更簡單
#更多的圖和caption
#venv312

from urllib.request import urlopen
from PIL import Image
from transformers import CLIPTokenizer, CLIPProcessor, CLIPModel
import torch
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
from PIL import Image


def main():
    model = SentenceTransformer('clip-ViT-B-32')
    
    puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
    cat_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/cat.png"
    car_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/car.png"
    paths = [puppy_path, cat_path, car_path]
    images = [Image.open(urlopen(path)).convert("RGB") for path in paths]
    # for image in images:
    #     image.show()

    captions = [
        "a puppy playing in the snow",
        "a pixelated image of a cute cat",
        "A supercar on the road with the sunset in the background",
    ]

    caption_embeddings = model.encode(captions)
    img_embeddings = model.encode(images)

    cos_scores = util.cos_sim(img_embeddings, caption_embeddings)
    print(cos_scores)


if __name__ == "__main__":
    main()