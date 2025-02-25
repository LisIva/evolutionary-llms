from evaluator import piped_evaluator, eq_buffer
from get_llm_response import get_response, get_debug_response
from rebuild_prompt import rebuild_prompt
from buffer_handler.eq_pruner import Pruner
from clean_directories import clean_output_dir, reset_prompt_to_init
from tqdm import tqdm
import sys
import traceback

MAX_ITER = 8
DIR_NAME = 'sindy-burg'
START_ITER = 0
REFINE_POINT = 100

DEBUG = True # True False
PRINT_EXC = True
EXIT = True


def perform_step(path, num, debug=False):
    if debug:
        response = get_debug_response(num=num)
    else:
        response = get_response(prompt_path=path, num=num, dir_name=DIR_NAME, print_info=False)
    score, eq_string, params = piped_evaluator(response, DIR_NAME)
    new_prompt, old_prompt = rebuild_prompt(eq_string, score, response, num=num, path=path)
    return new_prompt, score, eq_string, params


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

    for num in tqdm(range(START_ITER, min(REFINE_POINT, MAX_ITER)), desc="LLM's progress"):
        new_prompt, score, str_equation, params = step("prompts/continue-iter.txt", num, debug=DEBUG)

    # for num in tqdm(range(min(REFINE_POINT, START_ITER), MAX_ITER), desc="LLM's progress"):
    #     new_prompt, score, str_equation, params = step("prompts/continue-iter-refinement.txt", num, debug=DEBUG)

    optimization_track1 = eq_buffer
    pr = Pruner(eq_buffer.opt_track, 3)
    print()



