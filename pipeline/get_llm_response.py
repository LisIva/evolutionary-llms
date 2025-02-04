from openai import OpenAI
from pathlib import Path
from data.models_desc import models
import creds
from promptconstructor.combine_txts import read_with_langchain

MODEL = "qwen/qwen-2-72b-instruct"


def info(prompt, response, model_response):
    def cost(prompt, response):
        return len(prompt) / 1000 * models[MODEL]["in_price"] + \
               len(response) / 1000 * models[MODEL]["out_price"]

    system_prompt_price = len("You are a helpful assistant") / 1000 * models[MODEL]["in_price"]
    print(f"\n\nPrice: {cost(prompt, response):.5f}")
    print(f"Price with system prompt: {cost(prompt, response) + system_prompt_price:.5f}")
    print(f"Len(in_symbols): {len(prompt)}")
    print(f"Length of tokens, total: {model_response.usage.prompt_tokens + model_response.usage.completion_tokens}")
    print(f"Len(out_symbols): {len(response)}")


def get_debug_response(num=0):
    with open(f"debug_llm_outputs/out_{num}.txt") as debug:
        data = debug.read()
    return data


def get_response(prompt_path="prompts/continue-iter.txt", num=0, dir_name='burg', print_info=False):
    client = OpenAI(
        api_key=creds.api_key, base_url="https://api.vsegpt.ru/v1")

    prompt = read_with_langchain(path=prompt_path, dir_name=dir_name) # 2446 base len
    messages = [{"role": "user", "content": prompt}]
    response_big = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=1.0,
        n=1,
        max_tokens=2000, # максимальное число ВЫХОДНЫХ токенов
        extra_headers={ "X-Title": "EPDELLM"},)

    response = response_big.choices[0].message.content
    if print_info:
        print("Response:", response)
        info(prompt, response, response_big)
    with open(f"debug_llm_outputs/out_{num}.txt", 'w', encoding="utf-8") as model_out:
        model_out.write(response)
    return response


if __name__ == "__main__":
    get_response(print_info=True)