import numpy as np
import os
import sys
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator
np.set_printoptions(threshold=sys.maxsize)
# import tensorly as tl
# from tensorly.decomposition import tensor_train

PARENT_PATH = Path().absolute().parent
# дописать write_file для class Data


class Data(object):
    def __init__(self, dir_name, resample_shape=(20, 20)):
        self.dir_name = dir_name
        self.dir_path = os.path.join(PARENT_PATH, "data", "raw_derivs", dir_name)
        self.bounds = self.define_bounds(dir_name)
        self.raw_data, self.eval_data = self.load_raw()
        self.resampled_data = self.resample_data(resample_shape)

    def define_bounds(self, dir_name):
        if dir_name == 'burg': return ((0, 1), (-1000, 0))
        elif dir_name == 'wave': return ((0, 1), (0, 1))
        else:
            t = np.load(os.path.join(self.dir_path, "t.npy"))
            x = np.load(os.path.join(self.dir_path, "x.npy"))
            return ((np.min(t), np.max(t)), (np.min(x), np.max(x)))

    def load_raw(self):
        files = [file for file in os.listdir(self.dir_path) if file.endswith(".npy")]
        keys = []
        for file in files:
            key = file[:-4]
            for replaceable in (("x1", "t"), ("x2", "x"), ("_", "/")):
                key = key.replace(*replaceable)
            keys.append(key)

        raw_data = {}
        for i, file in enumerate(files):
            raw_data[keys[i]] = np.load(os.path.join(self.dir_path, file))

        if self.bounds is not None:
            t = np.linspace(self.bounds[0][0], self.bounds[0][1], raw_data['u'].shape[0])
            x = np.linspace(self.bounds[1][0], self.bounds[1][1], raw_data['u'].shape[1])
            grids = np.meshgrid(t, x, indexing='ij')
            raw_data['t'], raw_data['x'] = grids[0], grids[1]

        eval_data = {"inputs": [raw_data['t'], raw_data['x'], raw_data['u']]}
        derivs_dict = {}
        for key in list(raw_data.keys()):
            if key[0] == 'd':
                derivs_dict[key] = raw_data[key]
        eval_data['derivs_dict'] = derivs_dict
        return raw_data, eval_data

    def resample_data(self, shape):
        ti, xi = np.linspace(self.bounds[0][0], self.bounds[0][1], shape[0]), \
                 np.linspace(self.bounds[1][0], self.bounds[1][1], shape[1])
        grids_i = np.meshgrid(ti, xi, indexing='ij')
        test_points = np.array([grids_i[0].ravel(), grids_i[1].ravel()]).T
        interp_4u = RegularGridInterpolator([self.raw_data['t'][:, 0].ravel(), self.raw_data['x'][0, :]],
                                            self.raw_data['u'])
        u_res = interp_4u(test_points, method='linear').reshape(shape[0], shape[1])
        # resampled_data = {"inputs": [grids_i[0], grids_i[1], u_res]}
        resampled_for_truncation = {'t': grids_i[0].ravel(), 'x': grids_i[1].ravel(), 'u': u_res.ravel()}

        derivs_dict = {}
        for item in self.raw_data.items():
            if item[0][0] == 'd':
                interp = RegularGridInterpolator([self.raw_data['t'][:, 0].ravel(), self.raw_data['x'][0, :]],
                                            item[1])
                deriv_res = interp(test_points, method='linear').reshape(shape[0], shape[1])
                derivs_dict[item[0]] = deriv_res
                resampled_for_truncation[item[0]] = deriv_res.ravel()
        # resampled_data['derivs_dict'] = derivs_dict
        return resampled_for_truncation

    def write_resampled_data(self):
        file_name = f'{self.dir_name}_txu_derivs.txt'
        rounded_ls, names_data = self.round()
        with open(file_name, 'w') as myf:
            for aij in zip(*rounded_ls):
                line_ls = [f"{aij[k]}" for k in range(len(aij))]
                line = ' '.join(line_ls) + '\n'
                myf.write(line)
        # with open(file_name, 'w') as myf:
        #     for ti, xi, ui, u_ti, u_xi in zip(*rounded_ls):
                # myf.write(f'{ti} {xi} {ui} {u_ti} {u_xi}\n')

    def round(self):
        rounded_ls, names = [], []
        for item in self.resampled_data.items():
            arr = list(item[1])
            for i, el in enumerate(arr):
                if abs(el) > 100.:
                    arr[i] = np.round(el, 1)
                elif abs(el) > 10.:
                    arr[i] = np.round(el, 1)
                elif abs(el) > 1.:
                    arr[i] = np.round(el, 2)
                elif abs(el) > 0.0000001:
                    down_pow = int(np.abs(np.floor(np.log10(np.abs(el)))))
                    arr[i] = np.round(el, down_pow + 1)
                else: arr[i] = 0
            rounded_ls.append(arr)
            names.append(item[0])
        return rounded_ls, names


def load_resample_burg_array(name="u", shape=(20,20)):
    path = os.path.join(PARENT_PATH, "data", "simple_burg", f"{name}.npy")
    u = np.load(path)

    x, t = np.linspace(-1000, 0, 101), np.linspace(0, 1, 101)

    xi, ti = np.linspace(-1000, 0, shape[1]), np.linspace(0, 1, shape[0])
    grids = np.meshgrid(ti, xi, indexing='ij')
    test_points = np.array([grids[0].ravel(), grids[1].ravel()]).T

    interp = RegularGridInterpolator([t, x], u)
    u_res = interp(test_points, method='linear').reshape(shape[0], shape[1])
    return u_res,ti,xi, u


def write_file(t, x, u, u_t, u_x, show_symnum=False):
    # data = list(map(np.array2string, [t, x, u, u_t, u_x]))
    grids = np.meshgrid(t, x, indexing='ij')
    with open("burg_txu_derivs.txt", 'w') as myf:
        raveled_vals = list(map(np.ravel, [grids[0], grids[1], u, u_t, u_x]))
        for ti, xi, ui, u_ti, u_xi in zip(*raveled_vals):
            myf.write(f'{ti} {xi} {ui} {u_ti} {u_xi}\n')

    if show_symnum:
        with open("burg_txu_derivs.txt", 'r') as myf:
            content = myf.read()
            print("Количество символов:", len(content))


def local_round(t, x, u, u_t, u_x):
    t1 = np.round(t, 2)
    x1 = np.round(x, 1)
    u1 = np.round(u, 1)
    u_t1 = np.round(u_t, 1)
    u_x1 = np.round(u_x, 3)
    return t1, x1, u1, u_t1, u_x1


def get_simple_burg_data():
    u, t, x = load_resample_burg_array()
    u_t, _, _ = load_resample_burg_array("du_dx0")
    u_x, _, _ = load_resample_burg_array("du_dx1")
    t, x, u, u_t, u_x = local_round(t, x, u, u_t, u_x)
    write_file(t, x, u, u_t, u_x)


if __name__ == "__main__":
    u, t, x, u_f = load_resample_burg_array()
    u_t, _, _, u_t_f = load_resample_burg_array("du_dx0")
    u_x, _, _, u_x_f  = load_resample_burg_array("du_dx1")

    data_burg = Data('burg')
    # all_raveled_ls = []
    # for input in data_burg.eval_data['inputs']: # t, x, u
    #     # all_raveled_ls.append(np.expand_dims(input.ravel(), axis=0))
    #     all_raveled_ls.append(np.expand_dims(input, axis=2))
    #
    # for deriv_vals in data_burg.eval_data['derivs_dict'].values():
    #     all_raveled_ls.append(np.expand_dims(deriv_vals, axis=2))
    # input_mx = np.concatenate(all_raveled_ls, axis=2)
    # input_ten = tl.tensor(input_mx)
    #
    # original_shape = 101*101*5
    # cut_affordable = 30*30*5
    # cores = tensor_train(input_ten, rank=[1, 3, 3, 1])
    # reconstructed_tensor = tl.tt_to_tensor(cores)
    # error = np.sum(np.fabs(tl.to_numpy(input_ten) - tl.to_numpy(reconstructed_tensor))) / np.sum(input_ten) * 100
    # error_norm = np.linalg.norm(tl.to_numpy(input_ten) - tl.to_numpy(reconstructed_tensor)) / np.linalg.norm(tl.to_numpy(input_ten)) * 100
    #
    #
    # new_shape = cores[0].shape[0] * cores[0].shape[1] * cores[0].shape[2] + \
    #           cores[1].shape[0] * cores[1].shape[1] * cores[1].shape[2] + \
    #           cores[2].shape[0] * cores[2].shape[1] * cores[2].shape[2]
    # e2 = data_burg.eval_data['derivs_dict']


    # data_burg.write_resampled_data()
    data_burg_s = Data('sindy-burg')
    data_burg_s.write_resampled_data()

    data_burg_kdv = Data('kdv')
    # data_burg_kdv.write_resampled_data()

    data_burg_kdv_s = Data('sindy-kdv')
    # data_burg_kdv_s.write_resampled_data()

    data_wave = Data('wave')
    # data_wave.write_resampled_data()
    print()
    # t, x, u, u_t, u_x = local_round(t, x, u, u_t, u_x)
    # write_file(t, x, u, u_t, u_x, show_symnum=True)
    print()

