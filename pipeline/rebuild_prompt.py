import re
from extract_llm_response import retrieve_notes, compose_equation_v1_fun


def extract_exp_buffer(path, content=None):
    if content is None:
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


# def find_new_example_pos(file_content):
#     begin_pos = file_content.rfind('```python')
#     end_pos = file_content[begin_pos:].rfind('```') + begin_pos
#     return begin_pos, end_pos


def create_new_file(start_pos, end_pos, new_dict_str, response, continue_content, path, write_file=False, num=0):
    new_notes = retrieve_notes(response)
    new_str_fun = compose_equation_v1_fun(response)
    old_notes = retrieve_notes(continue_content)
    old_str_fun = compose_equation_v1_fun(continue_content)

    # if notes field exists replace 'def eq_1' and the notes
    if len(new_notes) > 8:
        continue_content = continue_content.replace(old_notes, new_notes)
        continue_content = continue_content.replace(old_str_fun, new_str_fun)
        continue_content = continue_content.replace("# An example of desired output:",
                                                    "# An example of desired output in the form of "
                                                    "previously discovered equation:")

    new_buff_file = continue_content[:start_pos] + new_dict_str + continue_content[end_pos:]

    if num == 0: path = "prompts/continue-iter.txt"
    if write_file:
        with open(path, 'w') as prompt:
            prompt.write(new_buff_file)
    return new_buff_file


def retrieve_copy_exp_buff(next_path, copy_from, num):
    start_pos, end_pos, dict_str, file_content = extract_exp_buffer(copy_from)
    start_pos_next, end_pos_next, dict_str_next, file_content_next = extract_exp_buffer(next_path)
    if start_pos_next == end_pos_next and num != 0:
        new_file_content = file_content_next[:start_pos_next] + dict_str + file_content_next[start_pos_next:]
        end_pos_next += len(dict_str)
        return start_pos_next, end_pos_next, dict_str, new_file_content
    elif num == 0:
        return start_pos, end_pos, dict_str, file_content
    else:
        return start_pos_next, end_pos_next, dict_str_next, file_content_next


def rebuild_prompt(insert_eq_str, value, response, path="prompts/continue-iter.txt", num=0):
    if len(insert_eq_str) > 250:
        raise Exception('The composed equation has an unaccepted structure: len(insert_eq_str) > 250')

    start_pos, end_pos, dict_str, file_content = retrieve_copy_exp_buff(next_path=path,
                                                                        copy_from="prompts/continue-iter.txt",
                                                                        num=num)
    if is_duplicate(insert_eq_str, dict_str):
        print(f'LLM generated a duplicate on iter #{num}')
        return None, None

    new_dict_str = insert_equation(insert_eq_str, value, dict_str)
    new_file = create_new_file(start_pos, end_pos, new_dict_str, response, file_content, path, write_file=True, num=num)
    return new_file, file_content


if __name__ == "__main__":
    #  new_file = rebuild_prompt("du/dt = c[0] * u", 0.8)
    #  new_file = rebuild_prompt("du/dt = c[0] * t + c[1] * x", 2)
    #  new_file = rebuild_prompt("du/dt = c[0] * u + c[1] * t", 1)
    #  new_file = rebuild_prompt("du/dt = c[0] * u + c[1] * du/dx", 300)
    # new_file = rebuild_prompt("du/dt = c[0] * u * du/dx + c[1] * t * du/dx", 1.65)
    print()
