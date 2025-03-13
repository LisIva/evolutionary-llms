class Record(object):
    def __init__(self, key, eq_code, loss, complex_score, eval_score, params, rs_code=None):
        self.key = key
        self.eq_code = eq_code
        self.loss = loss
        self.complex_score = complex_score
        self.eval_score = eval_score
        self.params = params
        self.rs_code = rs_code


class EqBuffer(object):
    def __init__(self):
        # doesn't contain subsets info:
        self.opt_track = {}
        self.records_track = {}

        # contain all equations, including subsets:
        self.full_opt_track = {}
        self.full_records_track = {}

    def push_record(self, key, complex_score, relat_score, loss, eq_code, params, rs_code):
        self.full_opt_track[key] = (complex_score, relat_score)
        self.opt_track[key] = (complex_score, relat_score)
        self.records_track[key] = Record(key, eq_code, loss, complex_score, relat_score, params, rs_code)
        self.full_records_track[key] = Record(key, eq_code, loss, complex_score, relat_score, params, rs_code)

    def push_subset_record(self, key, complex_score, relat_score, loss, eq_code, params):
        self.full_opt_track[key] = (complex_score, relat_score)
        self.full_records_track[key] = Record(key, eq_code, loss, complex_score, relat_score, params)
