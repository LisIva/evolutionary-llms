def find_eq_positions(response=None, path: str = 'out_0.txt', encoding: str = None):
    if response is None:
        with open(path, 'r', encoding=encoding) as myf:
            response = myf.read()

    begin_pos = response.rfind("def equation_v1(")
    if begin_pos == -1:
        undefined_begin = True
        begin_pos = response.find("right_side")
    else: undefined_begin = False

    return_pos = response[begin_pos:].find("return ") + begin_pos
    end_pos = response[return_pos:].find("\n")
    if end_pos == -1:
        end_pos = len(response) - 1
    else:
        end_pos += return_pos
    return begin_pos, end_pos, response, undefined_begin


# def move_end_cursor(begin_pos, end_pos, cut_code):
#     new_begin = end_pos + 1
#     end_pos = cut_code[new_begin-begin_pos:].find("\n")
#     pp = cut_code[new_begin-begin_pos:]


def get_code_part(eq_text, code_type='string_form_of_the_equation'):
    begin_pos = eq_text.find(f"{code_type} = ")
    line_end_pos = eq_text[begin_pos:].find("\n") + begin_pos
    # if line_end_pos-1 == '\\':
    #     cut_code = eq_text[begin_pos:]
    return eq_text[begin_pos + len(f"{code_type} = "):line_end_pos]


# can't replace '-' when multiline are in right_side
def replace_wrong_signs(eq_text, code_type='string_form_of_the_equation'):
    param = {'string_form_of_the_equation': 'c', 'right_side': 'params'}

    eq_origin = get_code_part(eq_text, code_type)
    eq_str = eq_origin.replace(f'-{param[code_type]}[0]', f'{param[code_type]}[0]', 1)

    if eq_str.find(f'- {param[code_type]}[') != -1:
        eq_str = eq_str.replace(f'- {param[code_type]}[', f'+ {param[code_type]}[')

    if eq_str.find(f'-{param[code_type]}[') != -1:
        eq_str = eq_str.replace(f'-{param[code_type]}[', f'+ {param[code_type]}[')

    if eq_str == eq_origin:
        return eq_text
    return eq_text.replace(eq_origin, eq_str)


def replace_wrong_coeffs(eq_text):
    str_old = get_code_part(eq_text, "string_form_of_the_equation")
    if str_old.find('{') == -1:
        return eq_text
    else:
        str_new = str_old.replace('{params', 'c')
        str_new = str_new.replace("{", "")
        str_new = str_new.replace("}", "")
        return eq_text.replace(str_old, str_new)


def add_tabulation(context):
    new_context = ['    ',]

    for i, char in enumerate(context):
        new_context += char
        if char == '\n':
            if i < len(context) - 2:
                if context[i+1] == ' ' and context[i+2] == ' ':
                    pass
                else: new_context += '    '

    new_context = ''.join(new_context)
    return new_context


def compose_equation_v1_fun(response=None, path='out_0.txt'):
    begin_pos, end_pos_newstr, context, undefined_begin = find_eq_positions(response=response, encoding="utf-8", path=path)
    if undefined_begin:
        tabbed_context = add_tabulation(context[begin_pos:end_pos_newstr+1])
        eq1_fun_text = 'def equation_v1(t, x, u, derivs_dict, params):\n' + tabbed_context
    else:
        eq1_fun_text = context[begin_pos:end_pos_newstr+1]

    eq1_fun_text = replace_wrong_coeffs(eq1_fun_text)
    eq1_fun_text = replace_wrong_signs(eq1_fun_text)
    eq1_fun_text = replace_wrong_signs(eq1_fun_text, code_type='right_side')
    return eq1_fun_text


def retrieve_notes(response=None, path: str = 'out_0.txt', encoding: str = None):
    if response is None:
        with open(path, 'r', encoding=encoding) as myf:
            response = myf.read()
    begin_pos = response.rfind('Important notes:')
    return_pos = response[begin_pos:].find('\n"""') + begin_pos
    notes = response[begin_pos:return_pos] + '\n"""'
    return notes


# def retrieve_example_response(response=None, path: str = 'out_0.txt', encoding: str = None):
#     if response is None:
#         with open(path, 'r', encoding=encoding) as myf:
#             response = myf.read()
#     begin_pos = response.rfind('```python')
#     return_pos = response[begin_pos:].rfind('```') + begin_pos
#     ex_response = response[begin_pos:return_pos]
#     return ex_response


if __name__ == "__main__":
    compose_equation_v1_fun(path='out_0.txt')
    # retrieve_example_response()
    # get_code_part('', code_type='right_side')


