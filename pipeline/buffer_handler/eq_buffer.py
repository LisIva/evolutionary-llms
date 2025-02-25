import numpy as np


def round_score(score):
    if score > 100:
        return int(score)
    elif score > 10:
        return np.round(score, 1)
    elif score > 1:
        return np.round(score, 2)
    else: return np.round(score, 3)


class Record(object):
    def __init__(self, key, eq_code, loss, complex_score, eval_score):
        self.key = key
        self.eq_code = eq_code
        self.loss = loss
        self.complex_score = complex_score
        self.eval_score = eval_score


class EqBuffer(object):
    def __init__(self):
        self.opt_track = {}
        self.records_track = {}

        self.__full_opt_track = {}

    def make_exp_buff(self):
        # exp_buffer = {}
        pass

    def push_record(self, key, complex_score, relat_score, loss, eq_code):
        self.__full_opt_track[key] = (complex_score, relat_score)
        self.opt_track[key] = (complex_score, round_score(relat_score))
        self.records_track[key] = Record(key, eq_code, loss, complex_score, round_score(relat_score))

    def reorder_by_knee(self):
        pass
