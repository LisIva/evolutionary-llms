import numpy as np
from pipeline.buffer_handler.sub_eq_builder import SubEqSet
from pipeline.buffer_handler.knee_reorder import SortedDict, KneeReorder, plot_track


class Pruner(object):
    def __init__(self, eq_buffer, evaluator, n_candidates, dir_name):
        self.dir_name = dir_name

        self.sorted_candid_track = SortedDict(eq_buffer.opt_track, sort_by=1)
        self.full_records_track = eq_buffer.full_records_track
        self.full_opt_track = eq_buffer.full_opt_track

        self.evaluator = evaluator
        self.eq_buffer = eq_buffer

        self.enriched_track = self.enrich_track(n_candidates)

    def enrich_track(self, n_candidates):
        candidates = self.sorted_candid_track.get_top_n(n_candidates)
        enriched_track = self.full_opt_track.copy()

        for candidate in candidates:
            parent_code = self.full_records_track[candidate[0]].rs_code

            eq_subset = SubEqSet(parent_code, candidate[0], self.dir_name,
                                 len(self.full_records_track[candidate[0]].params)).subset
            for sub_eq in eq_subset:
                if sub_eq not in self.full_records_track.keys():
                    complex_score, relat_score, params = self.evaluator.pruner_eval(sub_eq.feq_code, sub_eq.feq_str,
                                                                                    sub_eq.P, self.eq_buffer)
                    # self.eq_buffer.push_subset_record(sub_eq.feq_str, complex_score, relat_score, loss,
                    #                                   sub_eq.feq_code, params)
                    enriched_track[sub_eq.feq_str] = (complex_score, relat_score)
        return enriched_track

    def cut_by_knee(self):
        k_reorder = KneeReorder(self.enriched_track)
        pruned_track = k_reorder.projection_scores.sorted_dict
        by_project_track = k_reorder.by_projection.sorted_dict

        # k_reorder.knee_plot(plot_type='projection')
        return pruned_track, by_project_track


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