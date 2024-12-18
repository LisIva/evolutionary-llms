import re

BASE_COMPLEX_VAL = 0.4


def split_by_second_sign(terms_ls, sign='-'):
    terms = []
    for term in terms_ls:
        terms += term.split(sign)
    return terms


def glue_scattered(terms_ls):
    terms = []
    for i in range(len(terms_ls)-1):
        if terms_ls[i].count('(') == 0 and i == 0:
            terms.append(terms_ls[i])
        else: pass


def find_brace_idx(string):
    brace_open, brace_close = [], []
    for i in range(len(string)):
        if string[i] == '(':
            brace_open.append(i)
        elif string[i] == ')':
            brace_close.append(i)
    return brace_open, brace_close


def disassemble(string):
    terms_scattered = string.split(" = ")[1].split("+") # список из 1 или многих элементов
    terms_scattered1 = split_by_second_sign(terms_scattered)
    # br = re.findall('\(.*\)', string)
    terms = []
    for term in terms_scattered:
        if term.count('(') == 0:
            pass


def eval_fun(token):
    if token[:4] == 'du/d':
        return 1.
    elif token[:2] == 'd^':
        return (int(token[2]) + 1) * 0.5
    else: return BASE_COMPLEX_VAL


s2 = 'du/dt = a * t * exp(-1 + t * du/dx) - b * t * du/dx'
s3 = 'du/dt = -c[0] * (du/dx) - c[1] * t * x - c[2] * t + c[3]'
s4 = "du/dt = -c[0] * t - c[1] * du/dx"
s5 = f"du/dt = c[0] * du/dx + c[1] * t"
br = re.findall('[a-z]*\(.*\)', "u * du/dx * exp(1 + log(5x + u)) + c[2] * (du/dx)")
m6 = "du/dt = {c} * u * du/dx * exp(1 - log(5 * x + u)) + A[5] * (d^2u/dx^2) ^ 2 - {coeff} * t ^ 2 * u ^ 3"
# c1, c2 = find_brace_idx(m6)


def replace_pow(string):
    # replace pow ** with &
    string = string.replace('**', '&')

    # find pow indexes
    pow_idxs = [(m.start(0)+1, m.end(0)) for m in re.finditer('[^d][t-z]\^[0-9]+', string)]
    pow_idxs += [(m.start(0), m.end(0)) for m in re.finditer('\)\^[0-9]+', string)]
    pow_idxs = sorted(pow_idxs, key=lambda x: x[0])

    # form a new string by replacing a pow of ^ with &
    new_str = [string[:pow_idxs[0][0]]]
    for i in range(len(pow_idxs)):
        new_str.append(string[pow_idxs[i][0]:pow_idxs[i][1]].replace('^', '&'))
        if i != len(pow_idxs) - 1:
            new_str.append(string[pow_idxs[i][1]:pow_idxs[i+1][0]])
    new_str.append(string[pow_idxs[-1][1]:])

    return ''.join(new_str)


def clean_split_raw(string):
    string = string[8:]
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
        if j == 0:
            total_pow_val += eval_fun(s_ls5[i - 1]) * int(s_ls5[i][1:])
            processed_idxs += [i - 1, i]
        else:
            token, t_pow = s_ls5[i].split('&')
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
                total_val += 1.
                processed_ids.append(i)
            elif token[:2] == 'd^':
                total_val += (int(token[2]) + 1) * 0.5
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


m6_ls = clean_split_raw(m6)
total_pow_val, m6_ls = process_pow(m6_ls)
total_deriv_val, m6_ls = process_derivs(m6_ls)
m6_ls = remove_garbage(m6_ls)
total_val = BASE_COMPLEX_VAL * len(m6_ls) + total_pow_val + total_deriv_val
print()
