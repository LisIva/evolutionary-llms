import numpy as np
import matplotlib.pyplot as plt


def plot_track(opt_track, knee_scores):
    plt.grid()
    for point, name, knee_score in zip(opt_track.values(), opt_track.keys(), knee_scores.values()):
        x, y = point[0], point[1]
        mod = np.round(knee_score, 2)
        plt.scatter(x, y, s=200, edgecolors='k')
        plt.text(x + .03, y + .015, f'{name}: {mod}', fontsize=9)

    plt.legend(opt_track.keys())
    plt.show()


class SortedDict(object):
    def __init__(self, dictionary, sort_by=0, reverse=False):
        self.tupled_dict = type(list(dictionary.values())[0]) is tuple

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

    def get_max_idx(self, candidates):
        max_idx = 0
        for key, knee_score in candidates:
            idx = self.get_idx(key)
            if idx > max_idx:
                max_idx = idx
        return max_idx


class Point(object):
    def __init__(self, complex_score, eval_score, key):
        self.xy = np.array([complex_score, eval_score])
        self.name = key
        self.projection_score = None

    def set_projection_score(self, val):
        self.projection_score = val


class Vector(object):
    def __init__(self, p1, p2):
        self.coords = p2.xy - p1.xy

    def self_dot(self):
        return np.dot(self.coords, self.coords)


class KneeReorder(object):
    def __init__(self, opt_track):
        self.opt_track = opt_track

        self._by_complexity = SortedDict(opt_track, sort_by=0)
        self._by_eval = SortedDict(opt_track, sort_by=1)

        self.knee_scores = self.find_knee_scores()
        self._by_knee = SortedDict(self.knee_scores)

        self.end_point = self.find_end_point()
        self.start_point = self.find_start_point()
        self.projection_scores = self.calc_projection_scores()

    def find_knee_scores(self):
        knee_scores = {}
        for item in opt_track.items():
            knee_scores[item[0]] = self._by_complexity.max_val * (1 - item[1][1] / 1000) + 1.1 * item[1][0]
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

        end_point = Point(self.opt_track[end_point_key][0], self.opt_track[end_point_key][1], end_point_key)
        return end_point

    def find_start_point(self):
        key = list(self._by_knee.sorted_dict.items())[0][0]
        return Point(self.opt_track[key][0], self.opt_track[key][1], key)

    def calc_projection_scores(self):
        v = Vector(self.start_point, self.end_point)
        projection_scores = {}
        for item in self._by_complexity.sorted_dict.items():
            xi = Point(item[1][0], item[1][1], item[0])
            vi = Vector(self.start_point, xi)

            projection = np.dot(v.coords, vi.coords) / v.self_dot() * v.coords
            ri = vi.coords - projection

            if ri[0] < 0. and ri[1] < 0.:
                projection_scores[item[0]] = np.sqrt(np.sum(ri * ri)) # max ~= 3.5-4.0
        # proj_scores_sorted = SortedDict(projection_scores, reverse=True)
        return SortedDict(projection_scores, reverse=True)


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
                  'start_extr': (0.25, 1000.0),
                  'elbow_extr': (0.25, 1.0),
                  'end_extr': (3.7, 10.0)
                  }

    kn = KneeReorder(opt_track)
    plot_track(opt_track, kn.knee_scores)

    print()



