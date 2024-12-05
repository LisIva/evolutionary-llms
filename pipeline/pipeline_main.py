from evaluator import piped_evaluator
from get_llm_response import get_response, get_debug_response
from rebuild_prompt import rebuild_prompt
from clean_directories import clean_output_dir, reset_prompt_to_init
from tqdm import tqdm
import sys


MAX_ITER = 7
DEBUG = True
EXIT = False


def step(path, num=0, debug=False):
    try:
        if debug:
            response = get_debug_response(num=num)
        else:
            response = get_response(prompt_path=path, num=num)
        score, str_equation, params = piped_evaluator(response)
        new_prompt, old_prompt = rebuild_prompt(str_equation, score, num=num)

    except Exception as e:
        print(f"\nLLM generated an unacceptable response on iter #{num}:")
        if debug:
            print('Exception occured:', e, f'on iter#{num}\n')
            if EXIT:
                sys.exit()
        return None, None, None, None
    return new_prompt, score, str_equation, params


# LLM нашла бюргерса со второго раза, т к сначала предположила самую просутю зависимость: du/dt = k * du/dx
# надо описать это в начальном промпте: LLM должна знать что сначала следует генерить простые случаи и затем нанизывать на них ноые слагаемые
if __name__ == '__main__':
    # clean_output_dir()
    reset_prompt_to_init()

    new_prompt, score, str_equation, params = step("simple_burg_prompts/zero-iter.txt", 0, debug=DEBUG)
    for num in tqdm(range(1, MAX_ITER), desc="LLM's progress"):
        new_prompt, score, str_equation, params = step("simple_burg_prompts/continue-iter.txt", num, debug=DEBUG)
    print(str_equation)
