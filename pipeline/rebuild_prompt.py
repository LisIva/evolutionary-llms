import re


def extract_exp_buffer(path):
    with open(path) as prompt_file:
        content = prompt_file.read()
    start_str = "exp_buffer = {{\n"
    start_pos = content.find(start_str) + len(start_str)
    end_pos = content.find("}}", start_pos, len(content))
    dict_str = content[start_pos:end_pos]
    return start_pos, end_pos, dict_str, content


def is_duplicate(insert_eq_str, dict_str):
    examine_eq_str = "\"" + insert_eq_str + "\""
    groups = re.findall('(".*"): (.+),', dict_str)
    for (eq_str, score_str) in groups:
        if examine_eq_str == eq_str:
            return True
    return False


def insert_equation(insert_eq_str, insert_val, dict_str):
    if len(dict_str) == 0:
        return '"' + insert_eq_str + '"' + ": " + str(insert_val) + ","
    else:
        recreate_dict = []
        groups = re.findall('(".*"): (.+),', dict_str)
        for idx, (eq_str, score_str) in enumerate(groups):
            if insert_val < float(score_str): # insert_eq in dict body & end
                if idx < len(groups) - 1:
                    score_str_next = groups[idx + 1][1]
                    recreate_dict.append(eq_str + ": " + score_str)
                    if insert_val >= float(score_str_next):
                        recreate_dict.append('"' + insert_eq_str + '"' + ": " + str(insert_val))
                else:
                    recreate_dict.append(eq_str + ": " + score_str)
                    recreate_dict.append('"' + insert_eq_str + '"' + ": " + str(insert_val))
                    break
            elif insert_val >= float(score_str) and idx == 0: # insert_eq in dict start
                recreate_dict.append('"' + insert_eq_str + '"' + ": " + str(insert_val))
                recreate_dict.append(eq_str + ": " + score_str)
            else: recreate_dict.append(eq_str + ": " + score_str) # dict end
        return ",\n".join(recreate_dict) + ','


def create_new_buffer(start_pos, end_pos, new_dict_str, file_content, path, write_file=False):
    new_buff_file = file_content[:start_pos] + new_dict_str + file_content[end_pos:]
    if write_file:
        with open(path, 'w') as prompt:
            prompt.write(new_buff_file)
    return new_buff_file


def rebuild_prompt(insert_eq_str, value, path="simple_burg_prompts/continue-iter.txt", num=0):
    start_pos, end_pos, dict_str, file_content = extract_exp_buffer(path)
    if is_duplicate(insert_eq_str, dict_str):
        print(f'LLM generated a duplicate on iter #{num}')
        return None, None
    new_dict_str = insert_equation(insert_eq_str, value, dict_str)
    new_file = create_new_buffer(start_pos, end_pos, new_dict_str, file_content, path, write_file=True)
    # new_file = create_new_buffer(start_pos, end_pos, new_dict_str, file_content, path)
    return new_file, file_content


if __name__ == "__main__":
    #  new_file = rebuild_prompt("du/dt = c[0] * u", 0.8)
    #  new_file = rebuild_prompt("du/dt = c[0] * t + c[1] * x", 2)
    #  new_file = rebuild_prompt("du/dt = c[0] * u + c[1] * t", 1)
    #  new_file = rebuild_prompt("du/dt = c[0] * u + c[1] * du/dx", 300)
    # new_file = rebuild_prompt("du/dt = c[0] * u * du/dx + c[1] * t * du/dx", 1.65)
    print()
