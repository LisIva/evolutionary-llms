import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px


def plot_track(opt_track, knee_scores):
    plt.grid()
    for point, name, knee_score in zip(opt_track.values(), opt_track.keys(), knee_scores.values()):
        x, y = point[0], point[1]
        mod = np.round(knee_score, 2)
        plt.scatter(x, y, s=200, edgecolors='k')
        plt.text(x + .03, y + .015, f'{name}: {mod}', fontsize=9)

    # plt.legend(opt_track.keys())
    plt.show()


class SortedDict(object):
    def __init__(self, dictionary, sort_by=0, reverse=False):
        self.tupled_dict = type(list(dictionary.values())[0]) is tuple

        # sort_by: 0 - by complexity, 1 - by error
        if self.tupled_dict:
            self.sorted_dict = dict(sorted(dictionary.items(), key=lambda item: item[1][sort_by], reverse=reverse))
        else:
            self.sorted_dict = dict(sorted(dictionary.items(), key=lambda item: item[1], reverse=reverse))

        self.min_val, self.max_val = self.get_min_max_val(sort_by)

    def get_min_max_val(self, sort_by):
        sorted_ls = list(self.sorted_dict.values())
        if self.tupled_dict:
            val1, val2 = sorted_ls[-1][sort_by], sorted_ls[0][sort_by]
        else:
            val1, val2 = sorted_ls[-1], sorted_ls[0]
        return min(val1, val2), max(val1, val2)

    def get_idx(self, key):
        """Get the index of a specific key in the sorted dictionary."""
        items = list(self.sorted_dict.items())
        for idx, (k, _) in enumerate(items):
            if k == key:
                return idx
        return None

    def get_top_p(self, percent=0.3, top=True):
        n_candidates = int(np.round(percent * len(self.sorted_dict)))
        if top: return list(self.sorted_dict.items())[-n_candidates:]
        else: return list(self.sorted_dict.items())[:n_candidates]

    def get_top_n(self, n_candidates):
        return list(self.sorted_dict.items())[:n_candidates]

    def get_max_idx(self, candidates):
        max_idx = 0
        for key, knee_score in candidates:
            idx = self.get_idx(key)
            if idx > max_idx:
                max_idx = idx
        return max_idx


class Point(object):
    def __init__(self, complex_score, eval_score, key, transform=True):
        # without transform there is a high risk to have the best point with the eval_score of 400-500
        self.xy = np.array([complex_score, self.transform(eval_score)]) if transform \
                                                                        else np.array([complex_score, eval_score])
        self.name = key

    @staticmethod
    def transform(x):
        if x <= 200:
            return x
        elif 200 < x <= 400:
            return 1.5 * x
        else:
            return 2 * x


class Vector(object):
    def __init__(self, p1, p2):
        self.coords = p2.xy - p1.xy

    def self_dot(self):
        return np.dot(self.coords, self.coords)


class KneePlot(object):
    def __init__(self, opt_track: dict, knee_scores: dict,
                 projection_scores: SortedDict, end_point: Point, start_point: Point):
        self.knee_scores = knee_scores
        self.projection_scores = projection_scores.sorted_dict
        self.end_point = end_point
        self.start_point = start_point
        self.opt_track = opt_track

    def full_data_compile(self):
        x_axis, y_axis, labels, colors = [], [], [], []
        for item in self.opt_track.items():
            x_axis.append(item[1][0])
            y_axis.append(item[1][1])
            labels.append(f"{item[0]}: {np.round(self.knee_scores[item[0]], 2)}; "
                          f"{np.round(self.projection_scores.get(item[0], -1.), 2)}")
            if self.start_point.name == item[0] or self.end_point.name == item[0]:
                colors.append('red')
            else: colors.append('blue')
        data = {'complexity': x_axis, 'error': y_axis, 'info': labels, 'colors': colors}
        return pd.DataFrame(data)

    def project_data_compile(self):
        def append_info(key):
            x_axis.append(self.opt_track[key][0])
            y_axis.append(self.opt_track[key][1])
            labels.append(f"{key}: {np.round(self.knee_scores[key], 2)}; "
                          f"{np.round(self.projection_scores.get(key, -1.), 2)}")

        x_axis, y_axis, labels, colors = [], [], [], []
        for key in self.projection_scores.keys():
            append_info(key)
            colors.append('blue')

        for key in [self.start_point.name, self.end_point.name]:
            append_info(key)
            colors.append('red')

        data = {'complexity': x_axis, 'error': y_axis, 'info': labels, 'colors': colors}
        return pd.DataFrame(data)

    def plot_opt_track(self, plot_type):
        df = self.project_data_compile() if plot_type == 'projection' else self.full_data_compile()
        fig = px.scatter(df, x='complexity', y='error', text='info', color='colors', size=[0.1] * len(df))
        fig.update_traces(textposition='top center', marker=dict(size=14))
        fig.show()


class KneeReorder(object):
    def __init__(self, opt_track):
        self.opt_track = opt_track

        self._by_complexity = SortedDict(opt_track, sort_by=0)
        self._by_eval = SortedDict(opt_track, sort_by=1)

        self.knee_scores = self.find_knee_scores()
        self._by_knee = SortedDict(self.knee_scores)

        self.end_point = self.find_end_point()
        self.start_point = self.find_start_point()
        self.projection_scores, self.by_projection = self.calc_projection_scores()

    def knee_plot(self, plot_type='projection'):
        kp = KneePlot(self.opt_track, self.knee_scores, self.projection_scores, self.end_point, self.start_point)
        kp.plot_opt_track(plot_type)

    def find_knee_scores(self):
        knee_scores = {}
        for item in self.opt_track.items():
            knee_scores[item[0]] = self._by_complexity.max_val * (1 - item[1][1] / 1000) + item[1][0]
        return knee_scores

    def find_end_point(self):
        candidates = self._by_knee.get_top_p(0.3)
        max_idx = self._by_eval.get_max_idx(candidates)

        end_point_key, end_point_score = candidates[0][0], 0
        for key, knee_score in candidates:
            point_score = 1 - self._by_eval.get_idx(key)/max_idx + knee_score/self._by_knee.max_val
            if point_score > end_point_score:
                end_point_score = point_score
                end_point_key = key

        end_point = Point(self.opt_track[end_point_key][0], self.opt_track[end_point_key][1], end_point_key, False)
        return end_point

    def find_start_point(self):
        key = list(self._by_knee.sorted_dict.items())[0][0]
        return Point(self.opt_track[key][0], self.opt_track[key][1], key, False)

    def calc_projection_scores(self):
        v = Vector(self.start_point, self.end_point)
        projection_scores, by_projection = {}, {}
        for item in self._by_complexity.sorted_dict.items():
            xi = Point(item[1][0], item[1][1], item[0])
            vi = Vector(self.start_point, xi)

            projection = np.dot(v.coords, vi.coords) / v.self_dot() * v.coords
            ri = vi.coords - projection

            project_score = np.sqrt(np.sum(ri * ri))
            if ri[0] < 0. and ri[1] < 0.:# and item[1][1] < 400.:
                projection_scores[item[0]] = project_score # max ~= 3.5-4.0
                by_projection[item[0]] = project_score
            else:
                by_projection[item[0]] = -project_score
        # proj_scores_sorted = SortedDict(projection_scores, reverse=True)
        return SortedDict(projection_scores, reverse=True), SortedDict(by_projection, reverse=True)


if __name__ == '__main__':
    opt_track = {'c[0] * du/dx': (0.5, 1000.0),
                  'c[0] * (du/dx)**2 + c[1] * d^2u/dx^2': (1.75, 284.0),
                  'c[0] * d^2u/dx^2': (0.75, 282.0),
                  'c[0] * du/dx + c[1] * d^2u/dx^2': (1.25, 282.0),
                  'c[0] * t * du/dx': (0.7, 1000.0),
                  'c[0] * du/dx + c[1] * t * (du/dx)**2': (1.7, 982.0),
                  'c[0] * (du/dx)^2 + c[1] * d^2u/dx^2 + c[2] * (t + x)': (2.15, 285.0),
                  'c[0] * du/dx + c[1] * t ** 2': (0.9, 915.0),
                  '354.0; 2.4': (2.4, 354.0),
                  '400.0; 2.4': (2.4, 400.0),
                  '430.0; 2.25': (2.25, 430.0),
                  '950.0: 0.4': (0.4, 950.0),
                  '700.0: 0.4': (0.4, 700.0),
                  # 'start_extr': (0.25, 1000.0),
                  # 'elbow_extr': (0.25, 1.0),
                  # 'end_extr': (3.7, 10.0)
                  }

    kn = KneeReorder(opt_track)
    kn.knee_plot(plot_type='projection')
    # plot_track(opt_track, kn.knee_scores)

    print()



