def find_positions(response=None, path: str = 'llm-output.txt', encoding: str = None):
    if response is None:
        with open(path, 'r', encoding=encoding) as myf:
            response = myf.read()

    begin_pos = response.rfind("def equation_v1(")
    end_pos = response[begin_pos:].find("return (") + begin_pos
    return begin_pos, end_pos, response


# def replace_evaluate_code():
#     begin_pos, end_pos, context = find_positions(encoding="utf-8")
#     text_to_replace = context[begin_pos:end_pos]
#
#     begin_eval, end_eval, eval_file = find_positions('evaluator.py')
#     new_eval_file = eval_file[:begin_eval] + text_to_replace + eval_file[end_eval:]
#     exec(new_eval_file)


def write_equation_v1_fun(response=None):
    begin_pos, end_pos, context = find_positions(response=response, encoding="utf-8")
    end_of_fun_pos = context.find(")", end_pos, len(context))
    eq1_fun_text = context[begin_pos:end_of_fun_pos+1]
    return eq1_fun_text


if __name__ == "__main__":
    exec("def fun1(a, b):\n\treturn a+b\n")
    print(fun1(5, 6))
