import itertools
# from pipeline.extract_llm_response import get_code_part
import re
from promptconstructor.info_prompts import prompt_complete_inf


def sample_code(rs_code, eq_str, P):
    return f'def equation_v1(t: np.ndarray, x: np.ndarray, u: np.ndarray, derivs_dict: dict(), ' \
               f'params: np.ndarray):' \
               f'\n    right_side = {rs_code}' \
               f'\n    string_form_of_the_equation = "{eq_str}"\n    len_of_params = {P}' \
               f'\n    return right_side, string_form_of_the_equation, len_of_params'


def one_stroke_rs_code(eq_text):
    begin_pos = eq_text.find("right_side = ")
    eq_text = eq_text[begin_pos:]

    line_end_pos = eq_text.find("\n")
    # clean all redundant whitespaces ('\\', '\n', multiple ' '), return single-stroke format
    if eq_text[line_end_pos-1] == '\\':
        while eq_text[line_end_pos-1] == '\\':
            eq_text = eq_text[:line_end_pos-1] + eq_text[line_end_pos:]
            line_end_pos = eq_text[line_end_pos+1:].find("\n") + line_end_pos + 1
        rs_code = eq_text[len("right_side = "):line_end_pos]
        return re.sub('\s{2,}', ' ', rs_code)
    return eq_text[len("right_side = "):line_end_pos]


class Equation(object):
    def __init__(self, eq_str, eq_code, P, dir_name):
        left_deriv = prompt_complete_inf[dir_name]['left_deriv']
        self.feq_str = f'{left_deriv} = {eq_str}'
        self.feq_code = sample_code(eq_code, self.feq_str, P)


# не может перерабатывать уравнения, в которых c[..] названы иначе
class SubEqSet(object):
    def __init__(self, parent_code, parent_key, dir_name):
        self.dir_name = dir_name
        if parent_key[:len('string_form_of_the_equation = ')] == "string_form_of_the_equation = ":
            parent_key = parent_key[len('string_form_of_the_equation = '):]
        self.parent_terms_str = strip(split_with_braces(parent_key))

        rs_code = one_stroke_rs_code(parent_code)
        self.parent_terms_code = strip(split_with_braces(rs_code))
        self.subset = self.form_subset()

    def get_params_ids(self):
        c_ids = []
        params_ids = []
        total_num = 0
        for term_str, term_code in zip(self.parent_terms_str, self.parent_terms_code):
            c_matches = re.finditer(re.escape('c['), term_str)
            param_matches = re.finditer(re.escape('params['), term_code)
            params_ids.append([match.start() for match in param_matches])
            c_ids.append([match.start() for match in c_matches])

            total_num += len(c_ids[-1])
        return c_ids, params_ids, total_num

    def get_sub_eq_ids(self):
        parent_len = len(self.parent_terms_str)
        terms_idxs = [i for i in range(parent_len)]
        combin_idxs = []
        for r in range(1, parent_len):
            combin_idxs+=itertools.combinations(terms_idxs, r)
        return combin_idxs

    def form_subset(self):
        c_ids_all, params_ids_all, total_param_size = self.get_params_ids()
        comb_ids = self.get_sub_eq_ids()
        eq_subset = {}
        if total_param_size < 10: # otherwise can't reorder the terms properly
            for terms_id in comb_ids:
                # define elements of a new equation
                eq_str = [self.parent_terms_str[i] for i in terms_id]
                eq_code = [self.parent_terms_code[i] for i in terms_id]
                c_ids = [c_ids_all[i] for i in terms_id]
                params_ids = [params_ids_all[i] for i in terms_id]

                # rename the c[..] and params[..] by correct order and create an equation
                coef_reorder = CoeffReorder(eq_str, eq_code, c_ids, params_ids, total_param_size)
                eq_str, eq_code = coef_reorder.reorder_1digit()
                eq = Equation(' + '.join(eq_str), ' + '.join(eq_code), len(eq_code), self.dir_name)
                eq_subset[eq.feq_str] = eq.feq_code
        else:
            print("Total number of params exceeds 10, the equation can't have a subset")
        return eq_subset


class CoeffReorder(object):
    def __init__(self, eq_str: list, eq_code: list, c_ids: list, params_ids:list, total_param_size: int):
        self.eq_str, self.eq_code = eq_str, eq_code
        self.c_ids, self.params_ids = c_ids, params_ids
        self.total_param_size = total_param_size

    def rename_pos(self, i, idx, str_pos, code_pos, c_end=3, param_end=8):
        # len('c[') = 2, len('params[') = 7
        self.eq_str[i] = self.eq_str[i][:str_pos+2] + str(idx) + self.eq_str[i][str_pos + c_end:]
        self.eq_code[i] = self.eq_code[i][:code_pos+7] + str(idx) + self.eq_code[i][code_pos + param_end:]

    def reorder_1digit(self):
        idx = 0
        for i in range(len(self.eq_str)):
            for param_id, c_id in zip(self.params_ids[i], self.c_ids[i]):
                self.rename_pos(i, idx, c_id, param_id)
                idx += 1
        return self.eq_str, self.eq_code

    # def reorder_2digit(self):
    #     idx = 0
    #     for i in range(len(self.eq_str)):
    #         adhere_pos = 0
    #         for param_id, c_id in zip(self.params_ids[i], self.c_ids[i]):
    #             param_end = self.eq_code[i][param_id:].find(']')
    #             c_end = self.eq_str[i][c_id:].find(']')
    #
    #             self.rename_pos(i, idx, c_id-adhere_pos, param_id-adhere_pos, c_end, param_end)
    #             adhere_pos += c_end - 2 - len(str(idx))
    #             idx += 1


def strip(code_part_ls):
    new_code_part_ls = []
    for term in code_part_ls:
        new_code_part_ls.append(term.strip())
    return new_code_part_ls


def split_with_braces(code_part):
    '''
    :param code_str: right_side | eq_str variable in str format
    :return: list of str; splits the right_side string by '+', but only leaves the terms of the highest (braces) level
    '''
    terms = code_part.split('+')
    open_braces = terms[0].count('(') - terms[0].count(')')
    braces_exist = terms[0].count('(') > 0
    if open_braces != 0:
        i = 1
        while i < len(terms):
            term_br1 = terms[i].count('(')
            term_br2 = terms[i].count(')')
            if open_braces > 0:
                terms[i-1] += '+' + terms[i]
                terms.pop(i)
                i -= 1
            open_braces = open_braces + term_br1 - term_br2
            i += 1
    if braces_exist and len(terms) == 1:
        return split_with_braces(code_part[1:-1])
    return terms


# def get_char_idxs(string, char):
#     return [i for i, ltr in enumerate(string) if ltr == char]


eq_code = 'def equation_v1(t: np.ndarray, x: np.ndarray, u: np.ndarray, derivs_dict: dict(), params: np.ndarray):\n    right_side = params[0] * u * derivs_dict["du/dx"] + params[1] * derivs_dict["du/dx"] + params[2] * derivs_dict["d^2u/dx^2"]\n    string_form_of_the_equation = "du/dt = c[0] * u * du/dx + c[1] * du/dx + c[2] * d^2u/dx^2"\n    len_of_params = 3\n    return right_side, string_form_of_the_equation, len_of_params'
if __name__ == '__main__':
    # params[0] * u * derivs_dict["du/dx"] + params[1] * derivs_dict["du/dx"] + params[2] * derivs_dict["d^2u/dx^2"]
    # derivs_dict["du/dx"] ** 3 * t * params[0] + derivs_dict["du/dx"] * params[1]
    # c0*derivs_dict["du/dx"]**3*t + c1*derivs_dict["du/dx"]
    # c*derivs_dict["du/dx"]
    # params[0] * derivs_dict["du/dx"] + params[1] * t * (derivs_dict["du/dx"])**2
    # params[0] * derivs_dict["du/dx"] * t + params[1] * params[2]*x * derivs_dict["du/dx"]
    # (params[0] * t * derivs_dict["du/dx"] + params[1] * derivs_dict["du/dx"] + params[2] * x)
    # params[0] * t * derivs_dict["du/dx"] + params[1]
    # f"du/dt = {params[0]} * du/dx + {params[1]} * u + {params[2]} * d^2u/dx^2" for params[0] * derivs_dict["du/dx"] + params[1] * u + params[2] * derivs_dict["d^2u/dx^2"]

    #     du_dx = derivs_dict["du/dx"]
    #     d2u_dx2 = derivs_dict["d^2u/dx^2"]
    #     right_side = params[0] * du_dx + params[1] * d2u_dx2 + params[2] * t * du_dx
    rs_code = '(params[0] * (derivs_dict["du/dx"] + u + np.exp(x + t) *derivs_dict["du/dt"]) + t**2) + (params[1] * u * (t + x)) + (params[2] * derivs_dict["d^2u/dx^2"] * (t**2 + x**2)) + (params[3] * t**3) + (params[4] * x**3)'    # rs_code = 'params[0] * u * derivs_dict["du/dx"] + params[1] * derivs_dict["du/dx"] + params[2] * derivs_dict["d^2u/dx^2"]'
    eq_str = '(c[0] * (du/dx + u + exp(x + t) *du/dt) + t**2) + (c[1] * u * (t + x)) + (c[2] * d^2u/dx^2 * (t**2 + x**2)) + (c[3] * t**3) + (c[4] * x**3)'
    rs_code = rs_code.replace(' ', '')

    terms1 = split_with_braces(rs_code)
    terms2 = split_with_braces(eq_str)

    string_form_test1 = 'def equation_v1(t: np.ndarray, x: np.ndarray, u: np.ndarray, derivs_dict: dict(), params: np.ndarray):\n    right_side = t * derivs_dict["du/dx"] * params[0] + params[1] * x * derivs_dict["du/dx"] + params[2] * u * derivs_dict["d^2u/dx^2"]\n    string_form_of_the_equation = "du/dt = c[0] * t * du/dx + c[1] * x * du/dx + c[2] * u * d^2u/dx^2"\n    len_of_params = 3\n    return right_side, string_form_of_the_equation, len_of_params\n'

    string_form = 'def equation_v1(t: np.ndarray, x: np.ndarray, u: np.ndarray, derivs_dict: dict(), params: np.ndarray):\n    right_side = t * derivs_dict["du/dx"] * params[0] + \\\n                params[1] * x * derivs_dict["du/dx"] + \\\n                params[2] * u * derivs_dict["d^2u/dx^2"]\n    string_form_of_the_equation = "du/dt = c[0] * t * du/dx + c[1] * x * du/dx + c[2] * u * d^2u/dx^2"\n    len_of_params = 3\n    return right_side, string_form_of_the_equation, len_of_params\n'

    string_form_test2 = 'def equation_v1(t: np.ndarray, x: np.ndarray, u: np.ndarray, derivs_dict: dict(), params: np.ndarray):\n    right_side = t * derivs_dict["du/dx"] * params[0] * np.sin(params[1] * u) + params[2] * x * derivs_dict["du/dx"] + params[3] * u * derivs_dict["d^2u/dx^2"]\n    string_form_of_the_equation = "du/dt = c[0] * t * du/dx + c[1] * x * du/dx + c[2] * u * d^2u/dx^2"\n    len_of_params = 3\n    return right_side, string_form_of_the_equation, len_of_params\n'
    # sub_set2 = SubEqSet(string_form_test2, "t * du/dx * c[0] * sin(c[1] * u) + c[2] * x * du/dx + c[3] * u * d^2u/dx^2", 'burg')
    sub_set1 = SubEqSet(string_form_test1, "string_form_of_the_equation = t * du/dx * c[0] + c[2] * x * du/dx + c[3] * u * d^2u/dx^2", 'burg')

    # code10_1term = 'def equation_v1(t: np.ndarray, x: np.ndarray, u: np.ndarray, derivs_dict: dict(), params: np.ndarray):\n    right_side = t * derivs_dict["du/dx"] * params[10] + params[2] * x * derivs_dict["du/dx"] + params[3] * u * derivs_dict["d^2u/dx^2"] + params[0] * params[1] * params[4] * params[5] * params[6] * params[7] * params[8] * params[9]\n    string_form_of_the_equation = "du/dt = c[0] * t * du/dx + c[1] * x * du/dx + c[2] * u * d^2u/dx^2"\n    len_of_params = 3\n    return right_side, string_form_of_the_equation, len_of_params\n'
    # code10_2term = 'def equation_v1(t: np.ndarray, x: np.ndarray, u: np.ndarray, derivs_dict: dict(), params: np.ndarray):\n    right_side = t * derivs_dict["du/dx"] * params[10] * np.sin(params[11] * u) + params[2] * x * derivs_dict["du/dx"] + params[3] * u * derivs_dict["d^2u/dx^2"] + params[0] * params[1] * params[4] * params[5] * params[6] * params[7] * params[8] * params[9]\n    string_form_of_the_equation = "du/dt = c[0] * t * du/dx + c[1] * x * du/dx + c[2] * u * d^2u/dx^2"\n    len_of_params = 3\n    return right_side, string_form_of_the_equation, len_of_params\n'
    # str10_1term = "t * du/dx * c[10] + c[2] * x * du/dx + c[3] * u * d^2u/dx^2 + c[0] * c[1] * c[4] * c[5] * c[6] * c[7] * c[8] * c[9]"
    # str10_2term = "t * du/dx * c[10] * sin(c[11] * u) + c[2] * x * du/dx + c[3] * u * d^2u/dx^2 + c[0] * c[1] * c[4] * c[5] * c[6] * c[7] * c[8] * c[9]"
    # sub_set10 = SubEqSet(code10_2term, str10_2term, 'burg')
    print()