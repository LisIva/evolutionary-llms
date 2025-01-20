import re

BASE_COMPLEX_VAL = 0.2
BASE_DERIV_SCORE = 0.5


def split_by_second_sign(terms_ls, sign='-'):
    terms = []
    for term in terms_ls:
        terms += term.split(sign)
    return terms


def eval_fun(token):
    if token[:4] == 'du/d':
        return BASE_DERIV_SCORE
    elif token[:2] == 'd^':
        return (int(token[2]) + 1) * BASE_DERIV_SCORE / 2
    else: return BASE_COMPLEX_VAL


def replace_pow(string):
    # replace pow ** with &
    string = string.replace('**', '&')

    # find pow indexes
    pow_idxs = [(m.start(0)+1, m.end(0)) for m in re.finditer('[^d][t-z]\^[0-9]+', string)]
    pow_idxs += [(m.start(0), m.end(0)) for m in re.finditer('\)\^[0-9]+', string)]
    pow_idxs = sorted(pow_idxs, key=lambda x: x[0])

    # form a new string by replacing a pow of ^ with &
    if len(pow_idxs) != 0:
        new_str = [string[:pow_idxs[0][0]]]
        for i in range(len(pow_idxs)):
            new_str.append(string[pow_idxs[i][0]:pow_idxs[i][1]].replace('^', '&'))
            if i != len(pow_idxs) - 1:
                new_str.append(string[pow_idxs[i][1]:pow_idxs[i+1][0]])
        new_str.append(string[pow_idxs[-1][1]:])
        return ''.join(new_str)

    else: return string


def clean_split_raw(string):
    string = string[string.find('=')+2:]
    string = string.replace(' ', '')
    string = replace_pow(string)

    string = string.replace('{', '')
    string = string.replace('}', '')
    string = string.replace('[', '')
    string = string.replace(']', '')

    s_ls = string.split("+")
    s_ls1 = split_by_second_sign(s_ls, "-")
    s_ls2 = split_by_second_sign(s_ls1, "*")
    s_ls3 = split_by_second_sign(s_ls2, "(")
    s_ls4 = split_by_second_sign(s_ls3, ")")
    s_ls5 = [item for item in s_ls4 if
              item != '' and not item.isnumeric() and re.match('^[A-s]?$', item) is None and re.match('^[A-z][0-9]+$', item) is None]
    return s_ls5


def process_pow(s_ls5):
    terms_with_pow_idx = [(i, s_ls5[i].find("&")) for i in range(len(s_ls5)) if s_ls5[i].find("&") != -1]
    processed_idxs = []
    total_pow_val = 0.
    for i, j in terms_with_pow_idx:
        if j == 0: # '&' has idx = 0 and that means that i-1 is a complex term
            if re.match('^[0-9]+$', s_ls5[i][1:]) is not None:
                total_pow_val += eval_fun(s_ls5[i - 1]) * int(s_ls5[i][1:])
            else:
                total_pow_val += eval_fun(s_ls5[i - 1]) * 2.5
            processed_idxs += [i - 1, i]
        else:
            token, t_pow = s_ls5[i].split('&')
            if t_pow == '':
                total_pow_val += eval_fun(token) * 2.5
            else:
                total_pow_val += eval_fun(token) * int(t_pow)
            processed_idxs.append(i)
    s_ls5 = remove_processed_ids(s_ls5, processed_idxs[::-1])
    return total_pow_val, s_ls5


def remove_processed_ids(s_ls, processed_idxs):
    for idx in processed_idxs:
        s_ls.pop(idx)
    return s_ls


def process_derivs(s_ls):
    processed_ids = []
    total_val = 0.
    for i, token in enumerate(s_ls):
        if len(token) >= 5:
            if token[:4] == 'du/d':
                total_val += BASE_DERIV_SCORE
                processed_ids.append(i)
            elif token[:2] == 'd^':
                total_val += (int(token[2]) + 1) * BASE_DERIV_SCORE / 2
                processed_ids.append(i)
    s_ls = remove_processed_ids(s_ls, processed_ids[::-1])
    return total_val, s_ls


def remove_garbage(s_ls):
    remove_ids = []
    for i, token in enumerate(s_ls):
        if len(token) >= 4:
            remove_ids.append(i)
    s_ls = remove_processed_ids(s_ls, remove_ids[::-1])
    return s_ls


def eval_complexity(string):
    str_ls = clean_split_raw(string)
    total_pow_val, str_ls = process_pow(str_ls)
    total_deriv_val, str_ls = process_derivs(str_ls)
    str_ls = remove_garbage(str_ls)
    return BASE_COMPLEX_VAL * len(str_ls) + total_pow_val + total_deriv_val


if __name__ == '__main__':
    # hhhh = re.match('^[0-9]+$', '5241t')
    string = "du/dt = {c} * u * du/dx * exp(1 - log(5 * x + u)) + A[5] * (d^2u/dx^2) ^ 2 - {coeff} * t ^ 2 * u ^ 3"
    string_form_of_the_equation = "du/dt = c[0] * (du/dx)^3 + c[1] * (du/dx)^2 " + \
                              "+ c[2] * t * (u^2) * du/dx + c[3] * u * du/dx"
    print(len(string_form_of_the_equation))
    print()
