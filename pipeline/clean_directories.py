import os
import shutil


def clean_output_dir(dir_path="debug_llm_outputs/"):
    if len(os.listdir(dir_path)) == 0:
        print(f"The directory {dir_path} is already empty")

    for file_path in os.listdir(dir_path):
        try:
            full_path = os.path.join(dir_path, file_path)
            os.remove(full_path)
        except FileNotFoundError:
            print(f"The directory {dir_path} or doesn't exist")


def reset_prompt_to_init(dir_path="simple_burg_prompts/"):
    source = os.path.join(dir_path, "reset-for-continue.txt")
    target = os.path.join(dir_path, "continue-iter.txt")
    shutil.copyfile(source, target)


if __name__ == "__main__":
    clean_output_dir()
    reset_prompt_to_init()
