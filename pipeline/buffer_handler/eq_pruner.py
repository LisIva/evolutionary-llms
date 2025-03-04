import numpy as np
from pipeline.buffer_handler.sub_eq_builder import SubEqSet
from pipeline.buffer_handler.knee_reorder import SortedDict
# SortedDict


class Pruner(object):
    def __init__(self, eq_buffer, evaluator, n_candidates):
        self.sorted_candid_track = SortedDict(eq_buffer.opt_track, sort_by=1)
        self.full_records_track = eq_buffer.full_records_track
        self.full_opt_track = eq_buffer.full_opt_track
        self.evaluator = evaluator
        self.enrich_track(n_candidates)
        print()

    def enrich_track(self, n_candidates):
        candidates = self.sorted_candid_track.get_top_n(n_candidates)
        enriched_track = self.full_opt_track.copy()
        print()


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
    # f"du/dt = {params[0]} * du/dx + {params[1]} * u + {params[2]} * d^2u/dx^2" \foreq params[0] * derivs_dict["du/dx"] + params[1] * u + params[2] * derivs_dict["d^2u/dx^2"]

    #     du_dx = derivs_dict["du/dx"]
    #     d2u_dx2 = derivs_dict["d^2u/dx^2"]
    #     right_side = params[0] * du_dx + params[1] * d2u_dx2 + params[2] * t * du_dx

    rs_code = '(params[0] * (derivs_dict["du/dx"] + u + t *derivs_dict["du/dt"]) + t**2) + (params[1] * u * (t + x)) + (params[2] * derivs_dict["d^2u/dx^2"] * (t**2 + x**2)) + (params[3] * t**3) + (params[4] * x**3)'    # rs_code = 'params[0] * u * derivs_dict["du/dx"] + params[1] * derivs_dict["du/dx"] + params[2] * derivs_dict["d^2u/dx^2"]'
    # pruner = Pruner()
    print()