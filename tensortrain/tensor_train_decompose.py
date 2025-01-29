import numpy as np
import tensorly as tl
from tensorly.decomposition import tensor_train
from promptconstructor.array_to_txt import Data
import torch
np.random.seed(10)


def random_sample(data, shape=(3, 3)):
    flattened_input = data.flatten()
    random_indices = np.random.choice(flattened_input.size, size=shape[0] * shape[1], replace=False)
    sample = flattened_input[random_indices]
    return sample.reshape(shape)


class TT(object):
    def __init__(self, dir_name, rank=(1, 3, 3, 1), sample=False, resample_shape=(3, 3)):
        self.reconstructed_tensor = None
        self.dir_name = dir_name
        self.tt_rank = rank
        self.data = Data(dir_name)

        self.input_tensor = self.get_input_tensor(sample, resample_shape)
        self.cores = self.get_cores()

        self.relat_er_by_mae, self.relat_er_by_frob = self.get_reconstruction_error()
        self.tt_complexity = self.get_tt_complexity()

    def get_input_tensor(self, sample=False, shape=(3, 3)):
        data_ls = []
        for input in self.data.eval_data['inputs']:  # t, x, u
            # data_ls.append(np.expand_dims(input.ravel(), axis=0))
            if sample:
                input = random_sample(input, shape)

            data_ls.append(np.expand_dims(input, axis=2))

        for deriv_vals in self.data.eval_data['derivs_dict'].values():
            if sample:
                deriv_vals = random_sample(deriv_vals, shape)
            data_ls.append(np.expand_dims(deriv_vals, axis=2))

        input_mx = np.concatenate(data_ls, axis=2)
        return tl.tensor(input_mx)

    def get_cores(self):
        cores = tensor_train(self.input_tensor, rank=self.tt_rank)
        return cores

    def get_reconstruction_error(self):
        self.reconstructed_tensor = tl.tt_to_tensor(self.cores)
        # t11 = tt_to_tensor(self.cores)
        error = np.sum(np.fabs(tl.to_numpy(self.input_tensor) - tl.to_numpy(self.reconstructed_tensor))) / np.sum(
            self.input_tensor) * 100
        error_frobenius = np.linalg.norm(
            tl.to_numpy(self.input_tensor) - tl.to_numpy(self.reconstructed_tensor)) / np.linalg.norm(
            tl.to_numpy(self.input_tensor)) * 100
        return error, error_frobenius

    def get_tt_complexity(self):
        return self.cores[0].shape[0] * self.cores[0].shape[1] * self.cores[0].shape[2] + \
            self.cores[1].shape[0] * self.cores[1].shape[1] * self.cores[1].shape[2] + \
            self.cores[2].shape[0] * self.cores[2].shape[1] * self.cores[2].shape[2]


def tt_to_tensor(cores):
    ndim = len(cores)
    full_shape = [core.shape[1] for core in cores]

    # Initialize the contracted tensor with the first core reshaped
    current_tensor = cores[0].reshape(full_shape[0], -1)

    # Sequentially contract with remaining cores
    for i in range(1, ndim):
        core = cores[i]
        r_prev, n, r_next = core.shape
        core_reshaped = core.reshape(r_prev, n * r_next)

        # Contract along the shared mode
        current_tensor = np.dot(current_tensor, core_reshaped)

        # Reshape back to include the new dimension
        if i < ndim - 1:
            current_tensor = current_tensor.reshape(-1, n, r_next)

    # Final reshape to match the original tensor shape
    return current_tensor.reshape(full_shape)


def format_array(arr):
    lines = []
    new_line = ',\n '
    if arr.ndim == 3:
        for sub_array in arr:
            sub_lines = []
            for row in sub_array:
                formatted_row = ', '.join(f"{val:.3f}" if val > 0.1 else f"{val:.5f}" for val in row)
                sub_lines.append(f"[{formatted_row}]")
            lines.append(f"[{', '.join(sub_lines)}]")
        return f"[{new_line.join(lines)}]"

    for row in arr:
        formatted_row = ', '.join(f"{val:.3f}" if val > 0.1 else f"{val:.5f}" for val in row)
        lines.append(f"[{formatted_row}]")
    return f"[{new_line.join(lines)}]"


def write_array(array):
    # if array.shape[0] == 1:
    #     array = array.reshape((array.shape[-2], array.shape[-1]))
    # elif array.shape[2] == 1:
    #     array = array.reshape((array.shape[-3], array.shape[-2]))
    rounded_array = np.where(array > 0.1, np.round(array, 3), np.round(array, 5))
    formatted_array = format_array(rounded_array)
    with open('small_tt_train.txt', 'a') as f:
        f.write(formatted_array)
        f.write('\n\n')
    print(formatted_array)


def write_cores(train):
    cores = train.cores.factors
    write_array(cores[0])
    write_array(cores[1])
    write_array(cores[2])


if __name__ == '__main__':
    val = 20 * 20 * 7
    # kdv_tt = TT('kdv', (1, 4, 5, 1))
    kdv_tt = TT('kdv', (1, 2, 2, 1), sample=True, resample_shape=(2, 2))
    # write_cores(kdv_tt)
    U000 = 0.470118 + 0.017815 + 0.000469

    # Break down the reasoning step by step for each element of U and V. Output the tensor U first and ONLY then the reasoning!!!!
    # U = [
    #     [[0.976907, 0.385834, -0.003397, -0.004884, -0.027848, -0.106783, 0.175138],
    #      [1.181515, 0.342282, -0.012191, -0.004269, -0.026253, -0.105584, 0.178922],
    #      [1.132965, 0.321379, -0.010798, -0.004469, -0.025688,


    U = np.array([
        [
            [0.32132, 0.29798, -0.00279, -0.04221, 0.18208, -0.39787, 0.60529],
            [0.10635, 0.03876, -0.00876, -0.01993, 0.01592, -0.21452, 0.14739],
            [0.16326, 0.22495, -0.00179, -0.03288, 0.14094, -0.37326, 0.57983]
        ],
        [
            [0.24901, 0.21827, -0.00523, -0.03830, 0.16110, -0.34882, 0.52491],
            [0.11454, 0.04440, -0.00947, -0.02251, 0.01746, -0.20800, 0.14337],
            [0.17252, 0.23309, -0.00245, -0.03519, 0.14711, -0.36725, 0.57015]
        ],
        [
            [0.23909, 0.20786, -0.00505, -0.03700, 0.15836, -0.34412, 0.51915],
            [0.11166, 0.04225, -0.00923, -0.02140, 0.01699, -0.20426, 0.14072],
            [0.17016, 0.23033, -0.00234, -0.03420, 0.14529, -0.36418, 0.56669]
        ]
    ])

    # [[[0.19495, -0.32494, 0.01164, 0.14187, -0.01075, -0.01448, 0.14725],
    #   [0.30043, -0.10267, -0.16948, 0.02322, 0.00964, 0.00362, 0.01288],
    #   [0.01401, 0.00093, -0.01940, 0.00003, -0.00019, -0.00014, -0.00018]],
    #
    #  [[0.08468, -0.11048, 0.00427, 0.05324, -0.00408, -0.00556, 0.05693],
    #   [0.12673, -0.03870, -0.06543, 0.00928, 0.00389, 0.00147, 0.00522],
    #   [0.00567, 0.00039, -0.00706, 0.00001, -0.00008, -0.00006, -0.00007]],
    #
    #  [[0.01993, -0.02612, 0.00100, 0.01224, -0.00096, -0.00131, 0.01351],
    #   [0.03134, -0.00961, -0.01618, 0.00223, 0.00093, 0.00034, 0.00121],
    #   [0.00134, 0.00009, -0.00173, 0.00000, -0.00006, -0.00005, -0.00006]]]
    inp = kdv_tt.input_tensor
    # The calculations are done by taking the dot product of the corresponding dimensions in the tensors, following the rules of tensor contraction.
    # For the first step, we compute the dot product of the 3-dimensional vectors in G1 with the corresponding 3-dimensional vectors in G2,
    # for each of the 5 resulting vectors in V. For the second step, we compute the dot product of the resulting 5-dimensional vectors in V
    # with the corresponding 5-dimensional vectors in G3, for each of the 7 resulting vectors in U. This process is repeated for each of the 3x3 elements
    # in the final tensor U.

    # The calculations are done by taking the dot product of the corresponding dimensions in the tensors, following the rules of tensor contraction. For the first step, we compute the dot product of the 3-dimensional vectors in G1 with the corresponding 3-dimensional vectors in G2, for each of the 5 resulting vectors in V. For the second step, we compute the dot product of the resulting 5-dimensional vectors in V with the corresponding 5-dimensional vectors in G3, for each of the 7 resulting vectors in U. This process is repeated for each of the 3x3 elements in the final tensor U.
    # a = torch.arange(12.).reshape(3, 4)
    # b = torch.arange(24.).reshape(4, 3, 2)
    # b = torch.arange(40.).reshape(4, 5, 2)
    # t1 = torch.tensordot(a, b, ([1], [0]))
    # t1 = torch.tensordot(a, b)
    print()
