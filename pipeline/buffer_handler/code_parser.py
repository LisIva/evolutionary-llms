import re
from promptconstructor.info_prompts import prompt_complete_inf
import numpy as np
from sympy import expand, sympify


def strip(code_piece_ls):
    new_code_piece_ls = []
    for term in code_piece_ls:
        new_code_piece_ls.append(term.strip())
    return new_code_piece_ls


class EqSplitter(object):
    def __init__(self, code_piece):
        self.code_piece = code_piece
        self.terms = self.split_with_braces()

    def get_terms(self):
        return self.terms

    def split_with_braces(self):
        '''
        :param code_str: right_side | eq_str variable in str format
        :return: list of str; splits the right_side string by '+', but only leaves the terms of the highest (braces) level
        '''
        terms = self.code_piece.split('+')
        open_braces = terms[0].count('(') - terms[0].count(')')
        i = 1
        while i < len(terms):
            term_br1 = terms[i].count('(')
            term_br2 = terms[i].count(')')
            if open_braces > 0:
                terms[i - 1] += '+' + terms[i]
                terms.pop(i)
                i -= 1
            open_braces = open_braces + term_br1 - term_br2
            i += 1
        return terms

    def get_pretty_terms(self):
        for i, term in enumerate(self.terms):
            tokens = term.split('*')
            self.terms[i] = ' * '.join(tokens)

            tokens = term.split('+')
            self.terms[i] = ' + '.join(tokens)
        return self.terms


class RSVarsFixer(object):
    def __init__(self, rs_code, P):
        self.rs_code = rs_code
        self.P = P

    def fix(self):
        # try fixing parameters if they have wrong names
        if not self.has_correct_params():
            if self.P == 1:
                self.rs_code = self.rs_code.replace('c*', 'params[0]*')
            else:
                self.rs_code = self.fix_params()

        # try fixing derivs if they have wrong names
        if not self.has_correct_derivs():
            self.rs_code = self.fix_derivs()

        # if one of them is not correct raise exception
        if not self.has_correct_params() or not self.has_correct_derivs():
            raise Exception('Could not fix var names in rs_code variable')
        return self.rs_code

    def has_correct_params(self):
        return True if self.rs_code.find('params[') != -1 else False

    def has_correct_derivs(self):
        return True if self.rs_code.find('derivs_dict[') != -1 else False

    def fix_params(self):
        def replace_match(match):
            index = match.group(1)
            return f"params[{index}]"
        return re.sub(r"c\[(\d+)]", replace_match, self.rs_code)

    def fix_derivs(self):
        def replace_match(match):
            key = match.group(0)  # Extract the matched key (e.g., "du_dx", "d2u_dxn")

            # Handle first derivatives
            if key == "du_dx":
                return 'derivs_dict["du/dx"]'
            elif key == "du_dt":
                return 'derivs_dict["du/dt"]'

            # Handle higher-order derivatives
            elif key.startswith("d") and "_dx" in key:
                n = key.split("d")[1].split("u")[0]  # Extract the power (n)
                return f'derivs_dict["d^{n}u/dx^{n}"]'
            elif key.startswith("d") and "_dt" in key:
                n = key.split("d")[1].split("u")[0]  # Extract the power (n)
                return f'derivs_dict["d^{n}u/dt^{n}"]'

        pattern = r"d\d*u_d[xt]\d*"  # Matches "du_dx", "du_dt", "d2u_dx2", etc.
        return re.sub(pattern, replace_match, self.rs_code)


class RSExtractor(object):
    def __init__(self, eq_text, P=1, keep_header=False):
        self.eq_text = eq_text
        b, e = self.rs_code_boundary()

        self.cut_text = eq_text[b:e].replace(' ', '')
        first_line_pos = self.cut_text.find("\n")

        # extract rs_code according to its format
        if self.has_multiline(first_line_pos, self.cut_text):
            self.rs_code = self.extract_multiline(first_line_pos)
        elif self.has_split_code(first_line_pos, self.cut_text):
            self.rs_code = self.extract_split_code(first_line_pos)
        else:
            self.rs_code = self.extract_code(first_line_pos)

        self.rs_code = RSVarsFixer(self.rs_code, P).fix()

        if keep_header:
            self.rs_code = ''.join(['right_side=', self.rs_code])

    def rs_code_boundary(self):
        begin_pos = self.eq_text.find("right_side = ")
        end_pos = self.eq_text[begin_pos:].find('return ') + begin_pos
        return begin_pos, end_pos

    @staticmethod
    def has_multiline(first_line_pos, text):
        return text[first_line_pos - 1] == '\\'

    @staticmethod
    def has_split_code(first_line_pos, text):
        return True if text[first_line_pos:12+first_line_pos].find('right_side') > -1 else False

    def extract_multiline(self, line_pos):
        rs_code = self.cut_text
        while self.has_multiline(rs_code, line_pos):
            rs_code = rs_code[:line_pos - 1] + rs_code[line_pos + 1:]
            line_pos = rs_code[line_pos + 1:].find("\n") + line_pos + 1
        return rs_code[len("right_side="):line_pos]

    def extract_split_code(self, line_pos):
        if self.cut_text[line_pos:15 + line_pos].find('right_side+=') == -1:
            raise Exception('Unexpected split-code form in right_side variable')
        rs_code = self.cut_text
        while self.has_split_code(line_pos, rs_code):
            rs_code = rs_code[:line_pos] + '+' + rs_code[line_pos + 13:]
            line_pos = rs_code[line_pos + 1:].find("\n") + line_pos + 1
        return rs_code[len("right_side="):line_pos]

    def extract_code(self, line_pos):
        return self.cut_text[len("right_side="):line_pos]


class EqKeyReformer(object):
    def __init__(self, eq_key, P=1):
        # the ',' has no problem in itself, but subeq_set won't be able to work with it
        self.eq_key = eq_key.replace(' ', '')

        if self.eq_key.find(',') != -1:
            self.eq_key = self.eq_key[:self.eq_key.find(',')]

        if not self.has_correct_params():
            if P == 1:
                self.eq_key = self.fix_one_param()

        if not self.has_correct_params():
            raise Exception('Could not fix params in eq_key variable')

    def has_correct_params(self):
        return True if self.eq_key.find('c[') != -1 else False

    def fix_one_param(self):
        return self.eq_key.replace('c*', 'c[0]*').replace('k*', 'c[0]*')


class AssociativeBraces(object):
    def __init__(self, code_piece, brace_pairs):
        self.code_piece = code_piece
        self.pairs = brace_pairs

    def __check_start(self, s):
        if s-1 > 0 and (self.code_piece[s-1] == '+' or self.code_piece[s-1] == '('):
                return True
        elif s == 0: return True
        return False

    def __check_end(self, e):
        if e+1 < len(self.code_piece) and (self.code_piece[e+1] == '+' or self.code_piece[e+1] == ')'):
            return True
        elif e == len(self.code_piece) - 1:
            return True
        return False

    def open_braces(self):
        i = 0
        while i < len(self.pairs):
            if self.__check_start(self.pairs[i][0]) and self.__check_end(self.pairs[i][1]):
                self.__remove_braces(i)
            else:
                i += 1
        return self.code_piece, self.pairs

    def __subtract_value(self, number, i):
        if number < self.pairs[i][0]:
            return 0
        elif number > self.pairs[i][1]:
            return 2
        else:
            return 1

    def __update_pairs(self, i):
        indexes = [j for j in range(len(self.pairs)) if j != i]
        for j in indexes:
            self.pairs[j] = (self.pairs[j][0] - self.__subtract_value(self.pairs[j][0], i),
                             self.pairs[j][1] - self.__subtract_value(self.pairs[j][1], i))

    def __remove_braces(self, i):
        self.code_piece = self.code_piece[:self.pairs[i][0]] + self.code_piece[self.pairs[i][0] + 1:]
        self.__update_pairs(i)

        end_idx = self.pairs[i][1] - 1
        self.code_piece = self.code_piece[:end_idx] + self.code_piece[end_idx + 1:]
        self.pairs.pop(i)


class AssociativeBracesHandler(object):
    def __init__(self, eq_code_piece):
        self.eq_code_piece = eq_code_piece
        self.pairs, sort_len_pairs = self.find_pairs()

    def open_braces(self):
        if len(self.pairs) != 0:
            associative_braces = AssociativeBraces(self.eq_code_piece, self.pairs)
            return associative_braces.open_braces()
        else:
            return self.eq_code_piece, []

    def find_braces(self):
        starts = [match.start() for match in re.finditer(r'[(]', self.eq_code_piece)]
        ends = [match.start() for match in re.finditer(r'[)]', self.eq_code_piece)]
        return starts[::-1], ends

    def find_pairs(self):
        def find_pair(start, ends):
            for e in ends:
                if start - e < 0:
                    return e

        b_starts, b_ends = self.find_braces()
        b_pairs = []
        for s in b_starts:
            e = find_pair(s, b_ends)
            b_pairs.append((s, e))
            b_ends.remove(e)
        sorted_pairs = sorted(b_pairs, key=lambda x: x[1] - x[0], reverse=True)
        return b_pairs[::-1], sorted_pairs


class SympyConverter(object):
    def __init__(self, rs_code, params):
        self.rs_code = rs_code
        self.params = params

        self.trim_numpy()
        self.replace_derivatives()
        self.replace_params()
        self.sympy_code = expand(sympify(self.rs_code))

    def replace_params(self):
        def replace_match(match):
            index = match.group(1)  # Extract the number inside the brackets
            # return f"params{index}"
            return f"{self.params[int(index)]}"
        self.rs_code = re.sub(r"params\[(\d+)]", replace_match, self.rs_code)

    def trim_numpy(self):
        self.rs_code = self.rs_code.replace('np.', '').replace('numpy.', '')

    def replace_derivatives(self):
        def replace_match(match):
            key = match.group(1)  # Extract the key inside the quotes
            if key == "du/dx":
                return "du_dx"
            elif key == "du/dt":
                return "du_dt"

            # Replace higher-order derivatives
            elif "dx" in key:
                n = key.split("^")[1].split("u")[0]  # Extract the power (n)
                return f"d{n}u_dx{n}"
            else:
                n = key.split("^")[1].split("u")[0]  # Extract the power (n)
                return f"d{n}u_dt{n}"

        # Use re.sub with a pattern to match derivs_dict["..."]
        pattern = r'derivs_dict\["([^"]+)"\]'  # Matches derivs_dict["<key>"]
        self.rs_code = re.sub(pattern, lambda match: replace_match(match), self.rs_code)


# проверить все ли в порядке с edge case - т.к. сейчас парсер будет впихнут в конец оптимизации,
# т.е. в def eq должно быть ок для evaluator
if __name__ == '__main__':
    string_form_of_the_equation = "du/dt = k * du/dx , k = c[0]"
    eqk = EqKeyReformer(string_form_of_the_equation, 1).eq_key
    pop_track = {'du/dt = c[0] * du/dx + c[1] * t * du/dx + c[2] * t * x': (1.6, 460.5686610664196), 'du/dt = c[0] * du/dx + c[1] * u + c[2] * d^2u/dx^2': (1.45, 484.1114426561667), 'du/dt = c[0] * du/dx + c[1] * u * du/dx': (1.2, 438.94292729549943), 'du/dt = c[0] * u * du/dx + c[1] * du/dx + c[2] * d^2u/dx^2': (1.95, 37.14800565887713), 'du/dt = c[0] * u * du/dx + c[1] * d^2u/dx^2': (1.45, 38.90635312678824), 'du/dt = c[0] * u * du/dx + c[1] * d^2u/dx^2 + c[2] * du/dx * t': (2.15, 37.057907826954576), 'du/dt = c[0] * du/dx + c[1] * du/dt * d^2u/dx^2': (1.75, 542.9853705131861), 'du/dt = c[0] * du/dx + c[1] * t * du/dx': (1.2, 442.49077370655203)}
    rs11 = '    right_side = (params[0] * derivs_dict["du/dx"] ** 3\n    right_side += (params[1] * derivs_dict["du/dx"] ** 2 * np.cos(params[2] * u +1))\n    right_side += params[3] * derivs_dict["du/dt"] * t**2)\n    return right_side'
    rs12 = '    right_side = params[0] * derivs_dict["du/dx"] ** 3\n    right_side = right_side + params[1] * derivs_dict["du/dx"] ** 2\n    return right_side'
    rs2 = '    right_side = params[0] * derivs_dict["du/dx"] ** 3\\\n    + params[1] * derivs_dict["du/dx"] ** 2\\\n    + params[2] * derivs_dict["du/dt"]\n    return right_side'
    rs3 = '    right_side = params[0] * derivs_dict["du/dx"] ** 3 + params[1] * derivs_dict["du/dx"] ** 2\n    return right_side'
    rs4 = 'right_side = params[0] * derivs_dict["du/dx"] ** 3 + ((params[1] * derivs_dict["du/dx"] ** 2 * np.cos(params[2] * u +1)) * x**2) * u +params[3] * derivs_dict["du/dt"] * (t**2 + 2)\n    return right_side'
    rs5 = 'right_side = ((с[0]*du/dx + с[1]*u+1) + ((с[0]*du/dx&2*np.cos(с[1]*u+1))*x&2)*u)\n    return right_side'
    rs6 = 'right_side = ((с[0]*du/dx + с[1]*u+1) + ((с[0]*du/dx&2*np.cos((с[1]*u+1) + 6*u))*x&2)*u)\n    return right_side'

    rs7 = 'right_side = params[0] * np.exp(-params[1]*t) * derivs_dict["d^2u/dx^2"] + params[2]*u\n    return right_side'
    rs8 = 'right_side = params[0] * np.exp(t) * derivs_dict["d^2u/dx^2"]\n    return right_side'
    rs9 = 'right_side = c0 * np.exp(-c1*t) * derivs_dict["d^2u/dx^2"] + c2*u\n    return right_side'
    params1 = np.array([1.2678943, 5.898115, -7.264311])
    params2 = np.array([1.2678943, ])
    # cp = CodeParser(rs8, params2)
    print()

