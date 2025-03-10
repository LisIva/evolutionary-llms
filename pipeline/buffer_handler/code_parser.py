import re
from promptconstructor.info_prompts import prompt_complete_inf


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


def split_with_braces(code_part):
    '''
    :param code_str: right_side | eq_str variable in str format
    :return: list of str; splits the right_side string by '+', but only leaves the terms of the highest (braces) level
    '''
    terms = code_part.split('+')
    open_braces = terms[0].count('(') - terms[0].count(')')
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
    if terms[0].count('(') > 0 and len(terms) == 1:
        return split_with_braces(code_part[1:-1])
    return terms


def strip(code_part_ls):
    new_code_part_ls = []
    for term in code_part_ls:
        new_code_part_ls.append(term.strip())
    return new_code_part_ls


class RSExtractor(object):
    def __init__(self, eq_text, keep_header=False):
        self.eq_text = eq_text
        b, e = self.eq_code_boundary()

        self.cut_text = eq_text[b:e].replace(' ', '')
        first_line_pos = self.cut_text.find("\n")

        if self.has_multiline(first_line_pos, self.cut_text):
            self.eq_code = self.extract_multiline(first_line_pos)
        elif self.has_split_code(first_line_pos, self.cut_text):
            self.eq_code = self.extract_split_code(first_line_pos)
        else:
            self.eq_code = self.extract_code(first_line_pos)

        if keep_header:
            self.eq_code = ''.join(['right_side=', self.eq_code])

    def eq_code_boundary(self):
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


class BracesHandler(object):
    def __init__(self, eq_code):
        self.eq_code = eq_code
        self.b_starts, self.b_ends = self.find_braces()
        pairs = self.find_pairs()

    def find_braces(self):
        starts = [match.start() for match in re.finditer(r'[(]', self.eq_code)]
        ends = [match.start() for match in re.finditer(r'[)]', self.eq_code)]
        return starts[::-1], ends
    def __find_pair(self, start):
        for e in self.b_ends:
            if start - e < 0:
                return e

    def find_pairs(self):
        b_pairs = []
        for s in self.b_starts:
            e = self.__find_pair(s)
            b_pairs.append((s, e))
            self.b_ends.remove(e)
        return b_pairs


# class BracesHandler(object):
#     def __init__(self, eq_code):
#         self.eq_code = eq_code
#         self.terms = self.remove_redundant_braces(self.split_with_braces(eq_code))
#         self.open_braces()
#
#     def open_braces(self):
#         term_brace_start, term_brace_end = self.find_braces()
#         for i in range(len(term_brace_start)): # all braces inside a term
#             if term_brace_start[i][0] != -1:
#                 pass
                # i - индекс слагаемого, в term_brace_start[i] содержится тапл индексов переменной длины

    # def find_braces(self):
    #     brace_start, brace_end = [], []
    #     for term in self.terms:
    #         brace_idx1 = [match.start() for match in re.finditer(r'[(]', term)][::-1]
    #         if len(brace_idx1) != 0:
    #             brace_idx2 = [match.start() for match in re.finditer(r'[)]', term)]
    #             brace_start.append(tuple(brace_idx1))
    #             brace_end.append(tuple(brace_idx2))
    #         else:
    #             brace_start.append((-1,))
    #             brace_end.append((-1,))
    #     return brace_start, brace_end

    # def split_with_braces(self, eq_code):
    #     terms = eq_code.split('+')
    #     open_braces = terms[0].count('(') - terms[0].count(')')
    #     i = 1
    #     while i < len(terms):
    #         term_br1 = terms[i].count('(')
    #         term_br2 = terms[i].count(')')
    #         if open_braces > 0:
    #             terms[i - 1] += '+' + terms[i]
    #             terms.pop(i)
    #             i -= 1
    #         open_braces = open_braces + term_br1 - term_br2
    #         i += 1
    #     if terms[0][0] == '(' and terms[0][-1] == ')' and len(terms) == 1:
    #         return self.split_with_braces(eq_code[1:-1])
    #     return terms

    # @staticmethod
    # def has_redundant_braces(split_terms):
    #     for i in range(len(split_terms)):
    #         if split_terms[i][0] == '(' and split_terms[i][-1] == ')':
    #             return True
    #     return False
    #
    # def remove_redundant_braces(self, split_terms):
    #     new_split_terms = []
    #     for i in range(len(split_terms)):
    #         if split_terms[i][0] == '(' and split_terms[i][-1] == ')':
    #             new_split_terms += self.split_with_braces(split_terms[i][1:-1])
    #         else:
    #             new_split_terms.append(split_terms[i])
    #     if self.has_redundant_braces(new_split_terms):
    #         self.remove_redundant_braces(new_split_terms)
    #     return new_split_terms


class CodeParser(object):
    def __init__(self, eq_text):
        self.eq_text = eq_text
        eq_code = RSExtractor(eq_text).eq_code
        eq_code = eq_code.replace('**', "&")
        # code_parts = split_with_braces(eq_code)
        # eq_code = "((c[0] * t1 + c[1] * t2&2) + ((c[2] * t3 + t5) * x + t6))"
        bh = BracesHandler(eq_code)
        print()


if __name__ == '__main__':
    pop_track = {'du/dt = c[0] * du/dx + c[1] * t * du/dx + c[2] * t * x': (1.6, 460.5686610664196), 'du/dt = c[0] * du/dx + c[1] * u + c[2] * d^2u/dx^2': (1.45, 484.1114426561667), 'du/dt = c[0] * du/dx + c[1] * u * du/dx': (1.2, 438.94292729549943), 'du/dt = c[0] * u * du/dx + c[1] * du/dx + c[2] * d^2u/dx^2': (1.95, 37.14800565887713), 'du/dt = c[0] * u * du/dx + c[1] * d^2u/dx^2': (1.45, 38.90635312678824), 'du/dt = c[0] * u * du/dx + c[1] * d^2u/dx^2 + c[2] * du/dx * t': (2.15, 37.057907826954576), 'du/dt = c[0] * du/dx + c[1] * du/dt * d^2u/dx^2': (1.75, 542.9853705131861), 'du/dt = c[0] * du/dx + c[1] * t * du/dx': (1.2, 442.49077370655203)}
    rs11 = '    right_side = (params[0] * derivs_dict["du/dx"] ** 3\n    right_side += (params[1] * derivs_dict["du/dx"] ** 2 * np.cos(params[2] * u +1))\n    right_side += params[3] * derivs_dict["du/dt"] * t**2)\n    return right_side'
    rs12 = '    right_side = params[0] * derivs_dict["du/dx"] ** 3\n    right_side = right_side + params[1] * derivs_dict["du/dx"] ** 2\n    return right_side'
    rs2 = '    right_side = params[0] * derivs_dict["du/dx"] ** 3\\\n    + params[1] * derivs_dict["du/dx"] ** 2\\\n    + params[2] * derivs_dict["du/dt"]\n    return right_side'
    rs3 = '    right_side = params[0] * derivs_dict["du/dx"] ** 3 + params[1] * derivs_dict["du/dx"] ** 2\n    return right_side'
    rs4 = 'right_side = params[0] * derivs_dict["du/dx"] ** 3 + ((params[1] * derivs_dict["du/dx"] ** 2 * np.cos(params[2] * u +1)) * x**2) * u +params[3] * derivs_dict["du/dt"] * (t**2 + 2)\n    return right_side'
    rs5 = 'right_side = ((с[0]*du/dx + с[1]*u+1) + ((с[0]*du/dx&2*np.cos(с[1]*u+1))*x&2)*u)\n    return right_side'

    # right_side += params[2] * t * (u ** 2) * derivs_dict["du/dx"]
    # right_side += params[3] * u * derivs_dict["du/dx"]
    cp = CodeParser(rs5)
    print()

