from evaluator import piped_evaluator, optimization_track
from get_llm_response import get_response, get_debug_response
from rebuild_prompt import rebuild_prompt
from clean_directories import clean_output_dir, reset_prompt_to_init
from tqdm import tqdm
import sys
import traceback

MAX_ITER = 7
DIR_NAME = 'burg'
START_ITER = 2

DEBUG = False # True False
PRINT_EXC = True
EXIT = True


def perform_step(path, num, debug=False):
    if debug:
        response = get_debug_response(num=num)
    else:
        response = get_response(prompt_path=path, num=num, dir_name=DIR_NAME, print_info=False)
    score, str_equation, params = piped_evaluator(response, DIR_NAME)
    new_prompt, old_prompt = rebuild_prompt(str_equation, score, response, num=num)
    return new_prompt, score, str_equation, params


def step(path, num=0, debug=False):
    try:
        new_prompt, score, str_equation, params = perform_step(path, num=num, debug=debug)
    except Exception as e:
        print(f"\nException occurred on iter #{num}:")
        if PRINT_EXC:
            # EOL while scanning string literal
            print(traceback.format_exc())
        if EXIT:
            sys.exit()
        return None, None, None, None
    return new_prompt, score, str_equation, params


def step_0(path="prompts/zero-iter.txt", debug=False):
    new_prompt, score, str_equation, params = perform_step(path, num=0, debug=debug)
    return new_prompt, score, str_equation, params


if __name__ == '__main__':
    if START_ITER == 0:
        while True:
            try:
                if not DEBUG:
                    clean_output_dir()
                reset_prompt_to_init()
                new_prompt, score, str_equation, params = step_0(debug=DEBUG)
                START_ITER = 1
                break
            except Exception as e:
                print('An exception occurred on iter #0:')
                print(traceback.format_exc())
                if EXIT: sys.exit()

    for num in tqdm(range(START_ITER, MAX_ITER), desc="LLM's progress"):
        new_prompt, score, str_equation, params = step("prompts/continue-iter.txt", num, debug=DEBUG)

    optimization_track1 = optimization_track
    print()



