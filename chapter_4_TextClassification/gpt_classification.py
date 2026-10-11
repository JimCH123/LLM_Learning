import openai

client = openai.OpenAI(api_key= "API_KEY")
def chatgpt_generation(prompt, document, model="gpt-3.5-turbo-0125"):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a sentiment classifier. Respond only with 'positive' or 'negative'."},   #設定模型的整體角色/規則/行為
            {"role": "user", "content": prompt.replace("[DOCUMENT]", document)},
        ],
        temperature=0,
    )
    return response.choices[0].message.content    #response通常不是字串 而是很多欄位的回傳物件 可以透過print()查看

prompt = """Predict wheter the following document is a positive or negative movie review.:

[DOCUMENT]

If it is a positive review, please respond with "1". 
If it is a negative review, please respond with "0". 
Please do not give any other answer.
"""

document = "This movie is a great movie. The story is very interesting and the acting is superb. I would highly recommend it to anyone who enjoys a good film."

chatgpt_response = chatgpt_generation(prompt, document)

print(chatgpt_response)
