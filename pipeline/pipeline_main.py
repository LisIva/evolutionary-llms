from evaluator import piped_evaluator
from get_llm_response import get_response, get_debug_response
from rebuild_prompt import rebuild_prompt
from clean_directories import clean_output_dir, reset_prompt_to_init
from tqdm import tqdm
import sys


MAX_ITER = 5
DEBUG = True


def step(path, num=0, debug=False):
    try:
        if debug:
            response = get_debug_response(num=num)
        else:
            response = get_response(prompt_path=path, num=num)
        score, str_equation, params = piped_evaluator(response)
        new_prompt, old_prompt = rebuild_prompt(str_equation, score)

    except Exception as e:
        print(f"LLM generated an unacceptable response on iter #{num}:")
        if debug:
            print(e)
            sys.exit()
        return old_prompt, None, None, None
    return new_prompt, score, str_equation, params


# TODO: проверять уникальность уравнений на этапе переписывания промпта?
if __name__ == '__main__':
    # clean_output_dir()
    reset_prompt_to_init()

    new_prompt, score, str_equation, params = step("simple_burg_prompts/zero-iter.txt", 0, debug=DEBUG)
    for num in tqdm(range(1, MAX_ITER), desc="LLM's progress"):
        new_prompt, score, str_equation, params = step("simple_burg_prompts/continue-iter.txt", num, debug=DEBUG)
    print(str_equation)
