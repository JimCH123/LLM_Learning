# 需要啟用互動視窗 所以查看.ipynb檔案
from transformers import AutoProcessor, Blip2ForConditionalGeneration, BitsAndBytesConfig
from PIL import Image
from urllib.request import urlopen
import matplotlib.pyplot as plt
import torch
from IPython.display import HTML, display
import ipywidgets as widgets
import gc


def load_model(model_id):
    quant_config = BitsAndBytesConfig(load_in_8bit=True)  # 8-bit 量化省記憶體

    model = Blip2ForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quant_config,
        device_map="auto",
    )
    blip_processor = AutoProcessor.from_pretrained(model_id)
    return model, blip_processor

def main():

    gc.collect()
    torch.cuda.empty_cache()

    model_id = "Salesforce/blip2-opt-2.7b"
    car_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/car.png"
    image = Image.open(urlopen(car_path)).convert("RGB")
    # image.show()
    prompt = """
        Question : Write down what you see in this picture. 
        Answer: A sports car driving on the road at sunset.
        Question: What would it cost me to drive that car?
    """

    model, blip_processor = load_model(model_id)

    inputs = blip_processor(
        image, 
        text = prompt,
        return_tensors="pt",
    ).to(model.device, torch.float16)

    generated_ids = model.generate(**inputs, max_new_tokens=20)
    generated_text = blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    print(generated_text)


    memory = []
    output = widgets.Output()

    def text_eventhandler(change):
        output.append_stdout(f"[DEBUG] change = {change}\n")  # 用這個取代 print(change)
        question = change["new"]
        if not question:
            return
        change["owner"].value = ""

        if not memory:
            prompt = " Question: " + question + " Answer:"
        else:
            template = "Question: {} Answer: {}."
            prompt = " ".join(
                template.format(q, a) for q, a in memory
            ) + " Question: " + question + " Answer:"

        inputs = blip_processor(image, text=prompt, return_tensors="pt")
        inputs = inputs.to(model.device, torch.float16)

        generated_ids = model.generate(**inputs, max_new_tokens=100)

        # 關鍵修改:只取「新生成」的那段 token,不要把輸入的 prompt 部分也解碼進來
        input_len = inputs["input_ids"].shape[1]
        new_tokens = generated_ids[:, input_len:]

        generated_text = blip_processor.batch_decode(
            new_tokens,
            skip_special_tokens=True,
        )

        generated_text = generated_text[0].strip().split("Question")[0]

        memory.append((question, generated_text))

        output.append_display_data(HTML("<b>USER:</b> " + question))
        output.append_display_data(HTML("<b>BLIP-2:</b> " + generated_text))
        output.append_display_data(HTML("<br>"))

        
    in_text = widgets.Text()
    in_text.continuous_update = False
    in_text.observe(text_eventhandler, "value")


    # Display chat box
    display(
        widgets.VBox(
            children=[output, in_text],
            layout=widgets.Layout(display="inline-flex", flex_flow="column-reverse"),
        )
    )
        
    


if __name__ == "__main__":
    main()

