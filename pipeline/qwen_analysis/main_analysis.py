from openai import OpenAI
import creds
from promptconstructor.combine_txts import read_with_langchain


client = OpenAI(
    api_key=creds.api_key,
    base_url="https://api.vsegpt.ru/v1",
)

prompt = read_with_langchain(path='inp_analysis.txt', dir_name='burg')
print('The prompt length is:', len(prompt))

messages = [{"role": "user", "content": prompt}]
response_big = client.chat.completions.create(
    model="qwen/qwen-2-72b-instruct",
    messages=messages,
    temperature=1.0,
    n=1,
    max_tokens=1500, # максимальное число ВЫХОДНЫХ токенов
    extra_headers={ "X-Title": "My App"},
)

response = response_big.choices[0].message.content

print("Response:", response)
with open('out_analysis.txt', 'w') as myf:
    myf.write(response)
